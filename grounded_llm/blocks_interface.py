"""One-token pixel adapter using the shared model, loss and generation code."""
import json
import torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

from .interface import replace_vectors
from .training import losses
from .inference import generate_ids


class CompleteObject(StoppingCriteria):
    """Pause at syntactic JSON completion only; never inspect action legality."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        raw = self.tokenizer.decode(input_ids[0], skip_special_tokens=False).strip()
        if raw.endswith('}'):
            try:
                json.loads(raw)
                return True
            except ValueError:
                pass
        return False


class BlocksInterface:
    def __init__(self, config, model, tokenizer):
        self.config, self.model, self.tokenizer = config, model, tokenizer
        model.eval().requires_grad_(False)
        self.device = model.get_input_embeddings().weight.device
        self.dtype = model.get_input_embeddings().weight.dtype
        self.end_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.end_id
        self.slot = '<|blocks_current_state|>'
        tokenizer.add_tokens([self.slot], special_tokens=True)
        self.slot_id = tokenizer.convert_tokens_to_ids(self.slot)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def chat(self, text):
        return self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': text}], tokenize=True,
            add_generation_prompt=True, return_dict=False)

    def tensor_batch(self, sequences, states, adapter, answers=None):
        width = max(map(len, sequences))
        if width > self.config['generation']['context_limit']:
            raise ValueError('Context cap; no truncation')
        ids = torch.full((len(sequences), width), self.pad_id, dtype=torch.long, device=self.device)
        attention = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, sequence in enumerate(sequences):
            ids[i, -len(sequence):] = torch.tensor(sequence, device=self.device)
            attention[i, -len(sequence):] = 1
            if answers is not None:
                labels[i, -len(answers[i]):] = torch.tensor(answers[i], device=self.device)
        slots = ids == self.slot_id
        safe = ids.masked_fill(slots, self.pad_id)
        embeds = self.model.get_input_embeddings()(safe)
        if slots.any():
            if adapter is None:
                raise ValueError('State slot without adapter')
            pixels = [[int(x) for x in ''.join(rows)] for sample in states for rows in sample]
            vectors = adapter(torch.tensor(pixels, dtype=torch.float32, device=self.device))
            embeds = replace_vectors(embeds, slots, vectors)
        positions = (attention.cumsum(-1) - 1).masked_fill(attention == 0, 0)
        return dict(input_ids=ids, inputs_embeds=embeds, attention_mask=attention,
                    position_ids=positions, labels=labels)

    def batch(self, items, adapter, supervised=False):
        prompts = [self.chat(self.config['blocks']['report_prompt'] + '\nCurrent state: ' + self.slot)
                   for _ in items]
        answers = [self.encode(json.dumps(item['rows'], separators=(',', ':'))) + [self.end_id]
                   for item in items] if supervised else None
        sequences = [p + a for p, a in zip(prompts, answers)] if supervised else prompts
        return self.tensor_batch(sequences, [[item['rows']] for item in items], adapter, answers)

    def losses(self, items, adapter):
        return losses(self, items, adapter)

    @torch.no_grad()
    def generate_reports(self, items, adapter):
        batch = self.batch(items, adapter)
        generated = generate_ids(self.model, batch['inputs_embeds'], batch['attention_mask'],
                           self.generation_config(self.config['generation']['report_max_new_tokens']))
        return [self.decode(ids.tolist()) for ids in generated]

    def generation_config(self, budget):
        return GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=budget,
                                eos_token_id=self.end_id, pad_token_id=self.pad_id,
                                use_cache=True)

    def decode(self, ids):
        content = ids[:ids.index(self.end_id)] if self.end_id in ids else ids
        return self.tokenizer.decode(content, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)

    @torch.no_grad()
    def next_step(self, ids, states, adapter, budget):
        batch = self.tensor_batch([ids], [states], adapter)
        output = generate_ids(self.model, batch['inputs_embeds'], batch['attention_mask'],
                              self.generation_config(budget),
                              stopping_criteria=StoppingCriteriaList([CompleteObject(self.tokenizer)]))[0].tolist()
        return output, self.decode(output)

    def state_suffix(self, enabled):
        # Identical textual boundary in both groups; only the continuous slot differs.
        return self.encode('\nState:' + (self.slot if enabled else '') + '\nStep:\n')
