"""Conservative, persistent token-cost allowance for a user-run hosted stage.

This uses quoted rates, not a live provider wallet. Unknown/retried attempts
retain worst-case reservations; interruption cannot reset the allowance.
"""
from __future__ import annotations

import json
from pathlib import Path

from final_eval.inference.client import ModelResponse, generation_config
from final_eval.utils.artifacts import atomic_json, digest


class TokenCostGuard:
    def __init__(self, client, cfg, path, limit_inr, input_rate, output_rate):
        self.client, self.cfg, self.path = client, cfg, Path(path)
        self.input_rate, self.output_rate = input_rate, output_rate
        spec = {'generation':generation_config(cfg), 'limit_inr':limit_inr,
                'input_inr_per_million':input_rate, 'output_inr_per_million':output_rate,
                'input_bound':'UTF-8 prompt bytes plus1024 message/template tokens',
                'unknown_attempts':'reserve max_tokens at full output rate for every attempted call'}
        self.spec_hash = digest(spec)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'spec':spec,'spec_hash':self.spec_hash,'accounted_inr':0.0,
            'completed_generations':0,'unknown_attempts':0,'unsettled_reservations':[]}
        if self.state.get('spec_hash') != self.spec_hash:
            raise ValueError('Token-cost allowance settings changed; preserve the existing ledger')

    @property
    def attempts(self):
        return self.client.attempts

    def generate(self, prompt):
        # Byte-based BPE text input bound plus generous chat-template overhead.
        input_bound = len(prompt.encode('utf-8')) + 1024
        per_attempt = (input_bound*self.input_rate + self.cfg['max_tokens']*self.output_rate)/1e6
        reservation = per_attempt * int(self.cfg.get('max_attempts',2))
        if self.state['accounted_inr'] + reservation > self.state['spec']['limit_inr']:
            atomic_json(self.path,self.state)
            return ModelResponse(status='request_budget_exhausted',requested_model=self.cfg['name'],
                error='Token-cost allowance reached; no HTTP request sent. Preserve ledger and review usage.')
        pending = {'query_hash':digest(prompt),'reserved_inr':reservation,
                   'http_attempts_before':self.attempts}
        self.state['accounted_inr'] += reservation
        self.state['unsettled_reservations'].append(pending)
        atomic_json(self.path,self.state)
        # Exceptions/interruption deliberately leave the full reservation charged.
        response = self.client.generate(prompt)
        attempts = max(0,self.attempts-pending['http_attempts_before'])
        usage = response.usage or {}
        known = ('prompt_tokens' in usage and 'completion_tokens' in usage
                 and isinstance(usage['prompt_tokens'],int) and isinstance(usage['completion_tokens'],int)
                 and usage['prompt_tokens'] >= 0 and usage['completion_tokens'] >= 0)
        unknown = max(0,attempts-int(known))
        cost = ((usage['prompt_tokens']*self.input_rate + usage['completion_tokens']*self.output_rate)/1e6 if known else 0)
        cost += unknown*per_attempt
        self.state['accounted_inr'] += cost-reservation
        self.state['completed_generations'] += 1
        self.state['unknown_attempts'] += unknown
        self.state['unsettled_reservations'].remove(pending)
        atomic_json(self.path,self.state)
        return response
