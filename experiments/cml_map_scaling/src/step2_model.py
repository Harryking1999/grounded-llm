"""Frozen Qwen forward with differentiable continuous input slots."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache, GenerationConfig

from .step2_data import messages


class NodeIdGrammar:
    """Permit every node ID equally; never inspect edges, states, or answers."""
    def __init__(self, tokenizer, node_count):
        self.end_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.end_id
        self.allowed = {}
        for node in range(node_count):
            sequence = tokenizer.encode(str(node), add_special_tokens=False) + [self.end_id]
            for index, token in enumerate(sequence):
                self.allowed.setdefault(tuple(sequence[:index]), set()).add(token)

    def __call__(self, batch_id, generated):
        prefix = tuple(generated.tolist())
        if self.end_id in prefix:
            return [self.pad_id]
        return sorted(self.allowed[prefix])


def make_adapter(config, kind, hidden_size, device):
    torch.manual_seed(config['adapter']['initialization_seed'])
    dimension = config['assets']['state_dim']
    if kind == 'linear':
        adapter = nn.Linear(dimension, hidden_size, bias=config['adapter']['linear']['bias'])
    elif kind == 'mlp':
        spec = config['adapter']['mlp']
        adapter = nn.Sequential(nn.Linear(dimension, spec['hidden_dim'], bias=spec['bias']),
                                nn.GELU(), nn.Linear(spec['hidden_dim'], hidden_size, bias=spec['bias']))
    else:
        raise ValueError(kind)
    return adapter.to(device=device, dtype=torch.float32)


def answer_losses(hidden, labels, lm_head):
    """Shift once, project only supervised positions, average within each answer."""
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError('Every sample must have an answer including its end token')
    batch_indices = torch.arange(len(labels), device=labels.device)[:, None].expand_as(mask)[mask]
    logits = lm_head(hidden[:, :-1][mask]).float()
    token_losses = F.cross_entropy(logits, shifted_labels[mask], reduction='none')
    totals = torch.zeros(len(labels), device=hidden.device, dtype=token_losses.dtype)
    totals = totals.scatter_add(0, batch_indices, token_losses)
    return totals / counts


class StateInterface:
    def __init__(self, config, model, tokenizer, q, rms, adjacency):
        self.config, self.model, self.tokenizer = config, model, tokenizer
        self.device = model.get_input_embeddings().weight.device
        self.dtype = model.get_input_embeddings().weight.dtype
        self.q = torch.as_tensor(q / rms, dtype=torch.float32, device=self.device)
        self.adjacency = adjacency
        model.requires_grad_(False)
        model.eval()
        # These are tokenizer-only sentinels. They are replaced before lookup;
        # the LM vocabulary/weights are never resized or trained.
        slots = ['<|cml_current_state|>', '<|cml_goal_state|>']
        tokenizer.add_tokens(slots, special_tokens=True)
        self.slot_ids = [tokenizer.convert_tokens_to_ids(slot) for slot in slots]
        self.end_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.end_id
        if self.end_id is None:
            raise ValueError('Tokenizer must define the native assistant end token')
        self.prompt_cache = {}
        self.generation = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=config['generation']['max_new_tokens'],
            eos_token_id=self.end_id, pad_token_id=self.pad_id,
            bos_token_id=tokenizer.bos_token_id, use_cache=True)

    def prompt_ids(self, item, text_control):
        key = (item['u'] if text_control else None, item['g'] if text_control else None,
               item['task'], item['template'], text_control)
        if key not in self.prompt_cache:
            ids = self.tokenizer.apply_chat_template(
                messages(self.config, self.adjacency, item, text_control),
                tokenize=True, add_generation_prompt=True, return_dict=False)
            if not text_control:
                assert all(ids.count(slot) == 1 for slot in self.slot_ids)
            self.prompt_cache[key] = ids
        return self.prompt_cache[key]

    def batch(self, items, adapter, supervised=False, answer_override=None):
        text_control = adapter is None
        sequences, answers = [], []
        for index, item in enumerate(items):
            prompt = self.prompt_ids(item, text_control)
            answer = []
            if supervised:
                if item['task'] not in self.config['training']['tasks']:
                    raise ValueError('Action labels cannot be used for training')
                if item['task'] == 'action':
                    target = item['target']  # Only enabled by a separate action-supervision contract.
                else:
                    target = item['u'] if item['task'] == 'report_current' else item['g']
                text = str(target) if answer_override is None else answer_override[index]
                answer = self.tokenizer.encode(text, add_special_tokens=False) + [self.end_id]
            sequences.append(prompt + answer)
            answers.append(answer)
        length = max(map(len, sequences))
        if length > self.model.config.max_position_embeddings:
            raise ValueError('Input exceeds model context; truncation is forbidden')
        ids = torch.full((len(items), length), self.pad_id, dtype=torch.long, device=self.device)
        attention = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, (seq, answer) in enumerate(zip(sequences, answers)):
            ids[i, -len(seq):] = torch.tensor(seq, device=self.device)
            attention[i, -len(seq):] = 1
            if answer:
                labels[i, -len(answer):] = torch.tensor(answer, device=self.device)
        slot_masks = [ids == slot for slot in self.slot_ids]
        safe_ids = ids.clone()
        for mask in slot_masks:
            safe_ids[mask] = self.pad_id
        embeddings = self.model.get_input_embeddings()(safe_ids)
        if adapter is not None:
            for role, mask in zip(('u', 'g'), slot_masks):
                node_ids = torch.tensor([item[role] for item in items], device=self.device)
                vectors = adapter(self.q[node_ids]).to(self.dtype)
                embeddings = torch.where(mask[..., None], vectors[:, None, :], embeddings)
        positions = attention.cumsum(dim=-1) - 1
        positions.masked_fill_(attention == 0, 0)
        return dict(inputs_embeds=embeddings, attention_mask=attention,
                    position_ids=positions, labels=labels, input_ids=ids)

    def losses(self, items, adapter):
        batch = self.batch(items, adapter, supervised=True)
        output = self.model.model(inputs_embeds=batch['inputs_embeds'],
                                  attention_mask=batch['attention_mask'],
                                  position_ids=batch['position_ids'], use_cache=False)
        return answer_losses(output.last_hidden_state, batch['labels'], self.model.lm_head)

    @torch.no_grad()
    def generate(self, items, adapter, *, prefix_allowed_tokens_fn=None):
        batch = self.batch(items, adapter)
        # With inputs_embeds only, HF returns just the newly generated IDs.
        generated = self.model.generate(inputs_embeds=batch['inputs_embeds'],
                                        attention_mask=batch['attention_mask'],
                                        generation_config=self.generation,
                                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
        result = []
        for row in generated.tolist():
            ended = self.end_id in row
            answer_ids = row[:row.index(self.end_id)] if ended else row
            raw = self.tokenizer.decode(answer_ids, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
            result.append(dict(raw_output=raw, generated_ids=row,
                               native_end_seen=ended, hit_token_limit=not ended and len(row) >= self.generation.max_new_tokens))
        return result

    @torch.no_grad()
    def representation(self, adapter):
        return adapter(self.q).to(self.dtype).float().cpu().numpy()


class CachedStateInterface(StateInterface):
    """Cache only frozen causal prefix KV, retaining gradients through state slots.

    Used for report or action training on longer graphs. Generation is unchanged.
    The prefix ends immediately before the first learned state token, so no
    adapter-dependent activations are reused between optimizer steps.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.rebuild_prefix()

    def rebuild_prefix(self):
        """Refresh frozen KV when the visible graph changes, never across gradients."""
        item = dict(u=0, g=1, task='report_current', template='canonical')
        prompt = self.prompt_ids(item, False)
        self.prefix_length = prompt.index(self.slot_ids[0])
        self.prefix_ids = prompt[:self.prefix_length]
        with torch.no_grad():
            output = self.model.model(
                input_ids=torch.tensor([self.prefix_ids],device=self.device), use_cache=True)
        self.prefix_kv = [(layer[0].detach(),layer[1].detach()) for layer in output.past_key_values]

    def losses(self, items, adapter):
        if adapter is None or any(item['task'] not in ('report_current','report_goal','action') for item in items):
            raise ValueError('Frozen prefix caching requires latent graph tasks')
        batch = self.batch(items,adapter,supervised=True)
        suffixes, targets = [], []
        for i,item in enumerate(items):
            prompt = self.prompt_ids(item,False)
            if prompt[:self.prefix_length] != self.prefix_ids:
                raise ValueError('Report prompt does not share the frozen prefix')
            valid = batch['attention_mask'][i].bool()
            suffixes.append(batch['inputs_embeds'][i,valid][self.prefix_length:])
            targets.append(batch['labels'][i,valid][self.prefix_length:])
        length = max(len(s) for s in suffixes)
        embeddings = torch.stack([F.pad(s,(0,0,length-len(s),0)) for s in suffixes])
        labels = torch.stack([F.pad(t,(length-len(t),0),value=-100) for t in targets])
        attention = torch.ones((len(items),self.prefix_length+length),dtype=torch.long,device=self.device)
        for i,s in enumerate(suffixes):
            attention[i,self.prefix_length:self.prefix_length+length-len(s)] = 0
        positions = attention.cumsum(-1)[:,self.prefix_length:] - 1
        positions.masked_fill_(attention[:,self.prefix_length:]==0,0)
        # Each forward owns a fresh cache; update() must not append into the
        # saved prefix or retain another minibatch's autograd graph.
        cache = DynamicCache(config=self.model.config)
        for i,(key,value) in enumerate(self.prefix_kv):
            cache.update(key.expand(len(items),-1,-1,-1),value.expand(len(items),-1,-1,-1),i)
        output = self.model.model(inputs_embeds=embeddings,attention_mask=attention,
                                  position_ids=positions,past_key_values=cache,use_cache=True)
        return answer_losses(output.last_hidden_state,labels,self.model.lm_head)


# Existing scaling runs import this name; both names use the same implementation.
CachedReportInterface = CachedStateInterface
