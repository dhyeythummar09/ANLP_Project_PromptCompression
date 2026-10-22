# Copyright (c) 2023 Microsoft. MIT License.
# AST-extracted unchanged methods from official llmlingua 0.2.2 wheel.
# Offline regression fixture only; no pretrained weights or model loader.
# Wheel SHA256: da55137efe0db78063b3395396efe8a0dcfe4ae5a09aea0d503c34b7bf1d800c
from typing import List
import torch

class NativeFixture:

    def get_dynamic_compression_ratio(self, context: list, target_token: float, iterative_size: int, dynamic_ratio: list, start: int, seg_info: List[List[tuple]]=None):

        def get_ratio(base: float, delta: float):
            return max(min(1, base + delta), 0)
        context_length = [self.get_token_length(ii, False) + 2 for ii in context]
        if start:
            context_length = context_length[1:]
        tau = target_token / (sum(context_length) + 1)
        res, idx, last, last_target = ([], 0, 1, [])
        while idx < len(context_length):
            if last + context_length[idx] >= iterative_size:
                last_target.append((iterative_size - last, get_ratio(tau, dynamic_ratio[idx])))
                res.append(last_target)
                last = last + context_length[idx] - iterative_size
                if last > iterative_size:
                    k = last // iterative_size
                    res.extend([[(iterative_size, get_ratio(tau, dynamic_ratio[idx]))]] * k)
                    last -= k * iterative_size
                last_target = [(last, get_ratio(tau, dynamic_ratio[idx]))] if last else []
            else:
                last += context_length[idx]
                last_target.append((context_length[idx], get_ratio(tau, dynamic_ratio[idx])))
            idx += 1
        if last_target:
            res.append(last_target)
        return res

    def get_compressed_input(self, loss, input_ids, attention_mask, end=200, iterative_size=200, threshold=0.5, keep_flag=None, split_token_id: int=13, start: int=0, self_loss=None, self_input_ids=None, self_attention_mask=None):
        if self_loss is not None:
            need_idx = torch.concat([loss[:start] > 0, self_loss[:loss[start:].shape[0]] - loss[start:] > threshold, loss[:1] > 0])
        else:
            need_idx = torch.concat([loss > threshold, loss[:1] > 0])
        need_idx[end:] = 1
        need_idx[:end - iterative_size] = 1
        loss = loss[need_idx[:-1]]
        if self_loss is not None:
            if need_idx.shape[0] < self_loss.shape[0] + start + 1:
                need_idx = torch.cat([need_idx, torch.ones(self_loss.shape[0] - need_idx.shape[0] + start + 1, dtype=torch.bool).to(need_idx.device)])
            self_loss = self_loss[need_idx[start:-1]]
        if need_idx.shape[0] < input_ids.shape[1]:
            need_idx = torch.cat([need_idx, torch.ones(input_ids.shape[1] - need_idx.shape[0], dtype=torch.bool).to(need_idx.device)])
        elif need_idx.shape[0] > input_ids.shape[1]:
            need_idx = need_idx[:input_ids.shape[1]]
        if keep_flag is not None:
            need_idx[keep_flag == 1] = 1
        last = -1
        if keep_flag is not None:
            for ii in range(max(0, end - iterative_size), end):
                if need_idx[ii] != 1:
                    continue
                now = input_ids[0][ii].detach().cpu().item()
                if now == split_token_id and last == split_token_id and (keep_flag[ii].detach().cpu().item() == 0):
                    need_idx[ii] = 0
                else:
                    last = now
        compressed_input_ids = input_ids[attention_mask == 1][need_idx].unsqueeze(0)
        compressed_attention_mask = attention_mask[attention_mask == 1][need_idx].unsqueeze(0)
        if self_loss is not None:
            self_compressed_input_ids = self_input_ids[self_attention_mask == 1][need_idx[start:]].unsqueeze(0)
            self_compressed_attention_mask = self_attention_mask[self_attention_mask == 1][need_idx[start:]].unsqueeze(0)
        else:
            self_compressed_input_ids, self_compressed_attention_mask = (None, None)
        if keep_flag is not None:
            if len(keep_flag) > len(need_idx):
                keep_flag = torch.cat([keep_flag[:start], keep_flag[start:len(need_idx) + start][need_idx], keep_flag[start + len(need_idx):]])
            else:
                keep_flag = keep_flag[need_idx]
        end -= (need_idx[:end] == 0).sum()
        return (compressed_input_ids, compressed_attention_mask, keep_flag, end, loss, self_loss, self_compressed_input_ids, self_compressed_attention_mask)

    def get_estimate_threshold_base_distribution(self, ppl, ratio: float, condition_flag: bool=False):
        if ratio == 1.0:
            return float('-inf')
        ppl = ppl[ppl != 10000]
        target_token = max(0, min(len(ppl) - 1, int(len(ppl) * ratio) - 1))
        return ppl.sort(descending=not condition_flag).values[target_token].detach().cpu().item()

    def iterative_compress_prompt(self, context: List[str], target_token: float, iterative_size: int=200, keep_split: bool=False, split_token_id: int=13, start: int=0, dynamic_ratio: list=None, condition_compare: bool=False, segments_info: List[List[tuple]]=None):
        if segments_info is None or segments_info == []:
            iterative_ratios = self.get_dynamic_compression_ratio(context, target_token, iterative_size, dynamic_ratio, start)
        else:
            iterative_ratios = self.get_structured_dynamic_compression_ratio(context, iterative_size, dynamic_ratio, start, segments_info)
        context = '\n\n'.join(context)
        tokenized_text = self.tokenizer(context, return_tensors='pt', add_special_tokens=False)
        input_ids = tokenized_text['input_ids'].to(self.device)
        attention_mask = tokenized_text['attention_mask'].to(self.device)
        N = (attention_mask == 1).sum()
        compressed_input_ids, compressed_attention_mask = (input_ids, attention_mask)
        if condition_compare:
            self_input_ids, self_attention_mask = (input_ids[:, start:], attention_mask[:, start:])
            self_compressed_input_ids, self_compressed_attention_mask = (self_input_ids, self_attention_mask)
        end = min(iterative_size + start, compressed_input_ids.shape[1])
        threshold, keep_flag = (None, None)
        if keep_split:
            input_ids_numpy = input_ids.cpu().detach().numpy()[0]
            N = len(input_ids_numpy)
            keep_flag = [int(ii > 0 and input_ids_numpy[ii] == split_token_id and (input_ids_numpy[ii - 1] == split_token_id) or (ii < N - 1 and input_ids_numpy[ii] == split_token_id and (input_ids_numpy[ii + 1] == split_token_id))) for ii in range(N)]
            keep_flag = torch.tensor(keep_flag).to(self.device)
        past_key_values, past_loss, ready_end = (None, None, 0)
        self_past_key_values, self_past_loss, self_ready_end = (None, None, 0)
        pop_compressed_input_ids, pop_self_compressed_input_ids = (None, None)
        idx = 0
        while end <= compressed_input_ids.shape[1]:
            if end > self.max_position_embeddings and past_key_values is not None:
                e, s = (end - self.max_position_embeddings, min(self.cache_bos_num + start, self.max_position_embeddings))
                if pop_compressed_input_ids is None:
                    pop_compressed_input_ids = compressed_input_ids[:, :e]
                else:
                    pop_compressed_input_ids = torch.cat([pop_compressed_input_ids, compressed_input_ids[:, :e]], dim=-1)
                compressed_input_ids = compressed_input_ids[:, e:]
                compressed_attention_mask = compressed_attention_mask[:, e:]
                past_key_values = [[torch.cat([k[..., :s, :], k[..., s + e:, :]], dim=-2), torch.cat([v[..., :s, :], v[..., s + e:, :]], dim=-2)] for k, v in past_key_values]
                if keep_flag is not None:
                    keep_flag = keep_flag[e:]
                end, ready_end = (end - e, ready_end - e)
                if condition_compare:
                    s = min(s, self_past_key_values[0][0].shape[2] - e)
                    self_ready_end -= e
                    if pop_self_compressed_input_ids is None:
                        pop_self_compressed_input_ids = self_compressed_input_ids[:, :e]
                    else:
                        pop_self_compressed_input_ids = torch.cat([pop_self_compressed_input_ids, self_compressed_input_ids[:, :e]], dim=-1)
                    self_compressed_input_ids = self_compressed_input_ids[:, e:]
                    self_compressed_attention_mask = self_compressed_attention_mask[:, e:]
                    self_past_key_values = [[torch.cat([k[..., :s, :], k[..., s + e:, :]], dim=-2), torch.cat([v[..., :s, :], v[..., s + e:, :]], dim=-2)] for k, v in self_past_key_values]
            loss, past_key_values = self.get_ppl('', 'token', compressed_input_ids, compressed_attention_mask, past_key_values=past_key_values, return_kv=True, end=end if idx else None)
            if loss.shape[0] == 0:
                break
            if past_loss is not None:
                if end - 1 > len(past_loss):
                    past_loss = torch.cat([past_loss, torch.zeros_like(loss)[:end - 1 - len(past_loss)]])
                past_loss[ready_end:end - 1] = loss
                loss = past_loss
            else:
                past_loss = loss
            if idx:
                past_key_values = [[k[:, :, :end - iterative_size], v[:, :, :end - iterative_size]] for k, v in past_key_values]
            else:
                past_key_values = None
            if condition_compare:
                self_loss, self_past_key_values = self.get_ppl('', 'token', self_compressed_input_ids, self_compressed_attention_mask, past_key_values=self_past_key_values, return_kv=True, end=end - start if idx else None)
                if self_past_loss is not None:
                    if end - start - 1 > len(self_past_loss):
                        self_past_loss = torch.cat([self_past_loss, torch.zeros_like(self_loss)[:end - 1 - start - len(self_past_loss)]])
                    self_past_loss[self_ready_end:end - start - 1] = self_loss
                    self_loss = self_past_loss
                else:
                    self_past_loss = self_loss
                if idx:
                    self_past_key_values = [[k[:, :, :end - iterative_size - start], v[:, :, :end - iterative_size - start]] for k, v in self_past_key_values]
                else:
                    self_past_key_values = None
                self_ready_end = end - start - iterative_size if not (start and idx == 0) else 0
            ready_end = end - iterative_size if not (start and idx == 0) else 0
            for delta_end, ratio in iterative_ratios[idx]:
                loss = past_loss
                if condition_compare:
                    self_loss = self_past_loss
                    threshold = self.get_estimate_threshold_base_distribution(self_loss[:loss[start:].shape[0]] - loss[start:], ratio, False)
                else:
                    threshold = self.get_estimate_threshold_base_distribution(loss, ratio, False)
                compressed_input_ids, compressed_attention_mask, keep_flag, end, past_loss, self_past_loss, self_compressed_input_ids, self_compressed_attention_mask = self.get_compressed_input(loss, compressed_input_ids, compressed_attention_mask, end - iterative_size + delta_end, iterative_size=delta_end, threshold=threshold, keep_flag=keep_flag, split_token_id=split_token_id, start=start, self_loss=self_loss if condition_compare else None, self_input_ids=self_compressed_input_ids if condition_compare else None, self_attention_mask=self_compressed_attention_mask if condition_compare else None)
                end += iterative_size
            idx += 1
        if pop_compressed_input_ids is not None:
            compressed_input_ids = torch.cat([pop_compressed_input_ids, compressed_input_ids], dim=-1)
        return (compressed_input_ids[:, start:], compressed_attention_mask[:, start:])
