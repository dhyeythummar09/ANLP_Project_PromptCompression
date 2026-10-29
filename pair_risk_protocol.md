# V3 Pair-Risk Reranker — Protocol Design

> [!IMPORTANT]
> This is designed to slot directly into your existing `SpanScorer` + `compress.py` codebase. No new model class is needed. The pair scorer reuses the same DistilBERT backbone.

---

## Core Idea (one sentence)

Your current scorer gives each span an independent criticality score `p(span_i)`. The pair-risk reranker adds a **veto gate**: if two low-scored spans are about to be deleted together, and their pair interaction score is high, one of them is saved.

---

## Step 1 — Pair Selection Protocol

### What pairs to generate

For each question, you already have all its spans with:
- `span_category` (number, entity, negation, action_verb, comparison_conditional)
- `span_start`, `span_end` (character offsets in the prompt)
- `scorer_prob` from the v2 model (OOF predictions — these exist for training problems, and fold-model predictions for evaluation questions)

**Selection rule:** Pick pairs of spans that satisfy ALL of these:
1. Both spans have `scorer_prob < 0.5` (individually judged non-critical)
2. Spans are from **different categories** — same-category pairs (two numbers) are less likely to have interesting interactions than cross-category pairs (number + comparison_conditional, entity + action_verb)
3. Spans are **not adjacent** (start positions differ by at least 5 chars) — avoid testing trivially separate tokens
4. At most **K=5 pairs per question** — sample by highest combined score (both scorer_probs sum closest to 1.0, meaning both are near the decision boundary)

**Why `scorer_prob < 0.5` for both?** Because if either span is already marked critical by the v1 scorer, the compressor keeps it anyway. Pairs only matter when BOTH would be deleted.

### Estimated pair counts
- 160 evaluation questions × 5 pairs = **800 pairs**
- Each pair needs 4 conditions: original / delete-A / delete-B / delete-A+B
- Original is already known (use cached original answers from your existing `responses.csv`)
- So: 800 × 3 **new** reader calls = **2,400 calls** ← the real cost

> [!NOTE]
> You can reduce this to ~960 calls by reusing the 160 existing original responses. Only 800×3 = 2,400 calls remain — but if you use cached originals from your existing reader CSVs, you're at **2,400 net new calls**.

---

## Step 2 — Reader Call Script Design

Each condition sends the question with the indicated span(s) deleted (actual deletion, consistent with your Step 5 finding that mask ≈ delete 96.6% of the time):

```
uid = "bbh_date_understanding|bbh_date_understanding_21"
pair_id = "bbh_date_understanding_21__s47_e52__s89_e94"  # span positions

conditions:
  original   → full prompt (reuse from cache)
  delete_A   → delete chars [47:52]
  delete_B   → delete chars [89:94]
  delete_AB  → delete both [47:52] and [89:94]
```

### Label definition

```python
def is_risky_pair(original_correct, del_a_correct, del_b_correct, del_ab_correct):
    """
    A pair (A, B) is 'risky' if:
      - Deleting A alone is fine (model still correct)
      - Deleting B alone is fine (model still correct)
      - Deleting A+B together causes failure
    """
    return (
        original_correct and      # question is answerable at all
        del_a_correct and         # A alone: safe
        del_b_correct and         # B alone: safe
        not del_ab_correct        # A+B together: fails
    )
```

This is a **binary label per pair**: risky (1) or not risky (0).

Expected positive rate: probably 5–15% (most pairs that look safe individually ARE safe together). This is fine — it makes the task hard and interesting.

---

## Step 3 — Pair Scorer Architecture

### The key insight: you don't need a new model class

The existing `SpanScorer` does:
```
DistilBERT(question) → mean_pool(span_tokens) → linear → logit
```

The pair scorer does:
```
DistilBERT(question) → mean_pool(span_A_tokens) + mean_pool(span_B_tokens) → MLP → logit
```

Concretely, add this class to `train_scorer.py`:

```python
class PairRiskScorer(nn.Module):
    """
    Same DistilBERT encoder as SpanScorer.
    Takes two span masks per problem and predicts joint-deletion risk.
    Representation: [h_A || h_B || h_A - h_B || h_A * h_B]  (4 * hidden_size)
    """
    def __init__(self, encoder_name: str = MODEL_NAME, dropout: float = 0.1):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(encoder_name)
        H = self.encoder.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(4 * H, H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H, 1),
        )

    def forward(self, input_ids, attention_mask, mask_a, mask_b):
        """
        mask_a, mask_b: (B, L) float — token masks for span A and span B.
        Returns logits (B,).
        """
        h = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state  # (B, L, H)
        
        # Mean-pool each span
        h_a = (h * mask_a.unsqueeze(-1)).sum(1) / mask_a.sum(1, keepdim=True).clamp(1)
        h_b = (h * mask_b.unsqueeze(-1)).sum(1) / mask_b.sum(1, keepdim=True).clamp(1)
        
        # Interaction features: concat + difference + product
        feat = torch.cat([h_a, h_b, h_a - h_b, h_a * h_b], dim=-1)
        return self.classifier(self.dropout(feat)).squeeze(-1)
```

### Why this architecture?

- `h_A - h_B`: captures **asymmetry** (is A more critical given B is gone?)
- `h_A * h_B`: captures **co-occurrence** (are these the kind of spans that interact?)
- It's a 2-layer MLP, not just a linear — enough capacity to learn interactions, small enough to train in < 3 mins on T4

### Initialization trick (saves training time)

Load the v2 `scorer.pt` weights into `PairRiskScorer.encoder` before training:
```python
model = PairRiskScorer()
v2_state = torch.load("models/span_scorer_v2_20261021/fold0/scorer.pt")
# Load encoder weights only (classifier head is new)
model.encoder.load_state_dict({k.replace("encoder.", ""): v 
                               for k, v in v2_state.items() if k.startswith("encoder.")})
```
This warm-starts from v2's learned representations → faster convergence, fewer epochs needed (2 epochs instead of 4).

---

## Step 4 — Integration into compress.py

The compressor's current priority formula (from [`compress.py`](file:///Users/abhinavchatrathi/Documents/Sem6/ANLP_PROJ/ANLP_Project_PromptCompression/src/compressor/compress.py)):
```python
prio = a * prep["crit"] + (1 - a) * prep["info_rank"] + prep["prio_bonus"]
```

### How to plug in pair-risk

After computing individual priorities, add a post-processing **veto step**:

```python
def apply_pair_veto(words, prio, pair_risk_scores, veto_threshold=0.5):
    """
    For each risky pair (i, j) with pair_risk > threshold:
      Save the span with higher individual priority (lower deletion probability).
      Specifically: boost prio[j] if prio[i] < prio[j], else boost prio[i].
    Only activates when BOTH spans are below the deletion boundary.
    """
    for (i, j, risk_score) in pair_risk_scores:
        if risk_score > veto_threshold:
            # Save the higher-priority one by boosting it above the cut
            if prio[i] < prio[j]:
                prio[i] = max(prio[i], prio[j] + 0.1)  # keep i, let j be deleted
            else:
                prio[j] = max(prio[j], prio[i] + 0.1)
    return prio
```

This is **additive** — it doesn't replace the existing scorer, it just protects the most valuable span in a risky pair.

---

## Step 5 — What to evaluate

After retraining + recompressing:

| Metric | Compare |
|--------|---------|
| Downstream accuracy | `ours_v2` vs `ours_v3` (pair-aware) at 0.50 and 0.75 |
| Risky pair rate | How many pairs in the test set were vetoed? |
| Veto effectiveness | When a pair was vetoed, did it actually help? (del_AB failure rate on vetoed vs non-vetoed) |
| Compression ratio | Does vetoing change actual token retention? (should be minor — ≤1 span saved per question) |

---

## Team Division of Labor (parallel track)

```
Hour 0–2:
  Person A  → Write pair_selector.py: takes step4_input.csv + no_bonuses_compressed scores
              → outputs pairs CSV with (uid, span_a_start, span_a_end, span_b_start, span_b_end)
              
  Person B  → Write pair_reader_calls.py: takes pairs CSV → sends 3 conditions per pair to Qwen
              → outputs pair_labels.csv (risky: True/False per pair)
              
  Person C  → Add PairRiskScorer class to train_scorer.py (architecture above)
              → Write train_pair_scorer.py: load pairs, 5-fold, warm-start from v2 weights
              
  Person D  → Write followup_summary.md (steps 3–5 unified) + date-understanding subgroup
              (this is independent, needed regardless)

Hour 2–4:
  A + B run Jarvislabs: 2,400 reader calls (split 1,200 each) → ~1.5 hrs

Hour 4–5:
  C: merge labels → train PairRiskScorer on Kaggle T4 (~3–5 min per fold)
  B: patch compress.py with apply_pair_veto
  A: design evaluation script

Hour 5–7:
  All: re-compress 160 questions with pair-aware scorer
  Run 160-Q reader eval on Jarvislabs (~1 hr)
  Analyze + write up
```

---

## Honest risk assessment

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Positive rate too low (<5%) | Medium | The pair veto still shows up as a controlled result — "we found few joint failures" is publishable |
| Accuracy doesn't improve over v2 | Medium-High | Still publishable — confirms single-span scoring captures most of the signal |
| Pair scorer overfits (too few positive examples) | Medium | Warm-start from v2 + strong regularization (dropout 0.2, weight decay 0.1) |
| API cost exceeds budget | Low | ~2,400 calls × ~200 tokens avg = 480K tokens; at Qwen's rate ~INR 136 — well within budget |

> [!TIP]
> Even if the pair-risk reranker doesn't improve accuracy, the **finding itself is valuable**: "individually safe deletions rarely fail jointly (only X% of pairs)" directly answers a question about whether your single-span labeling assumption is sufficient. Combined with Step 5's 96.6% mask/delete agreement, this builds a complete validation story.
