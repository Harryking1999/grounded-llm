"""Authority for frozen Q state lookup, adapters and continuous embedding slots."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache, GenerationConfig
from .prompts import messages, augmentation_messages, build_prompt, decision_messages

def state_interface(q, node_ids, adapter, dtype):
    """Read true state Q rows and map them to continuous model embeddings."""
    if q is None:
        raise ValueError('State vectors require a map asset')
    return adapter(q[node_ids]).to(dtype)


def replace_vectors(embeddings, mask, vectors):
    """Replace explicit state slots; supports graph lookup and raw pixel states."""
    if int(mask.sum()) != len(vectors):
        raise ValueError('State slot count does not match supplied vectors')
    result = embeddings.clone()
    result[mask] = vectors.to(embeddings.dtype)
    return result

def inject_vectors(embeddings, ids, slots, items, q, adapter, dtype, permutation=None):
    """One authority for vector injection, independent of prompt and task labels."""
    if adapter is None:
        return embeddings
    for role, slot in slots.items():
        mask = ids == slot
        if not mask.any():
            continue
        if torch.any(mask.sum(dim=1) > 1):
            raise ValueError('Each state role may appear at most once per example')
        nodes = [item.get(role, 0) for item in items]
        if permutation is not None:
            nodes = [permutation[node] for node in nodes]
        vectors = state_interface(q, torch.tensor(nodes, device=ids.device), adapter, dtype)
        embeddings = replace_vectors(embeddings, mask, vectors[mask.any(dim=1)])
    return embeddings

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

class StateInterface:
    def __init__(self, config, model, tokenizer, q, rms, adjacency):
        self.config, self.model, self.tokenizer = config, model, tokenizer
        self.device = model.get_input_embeddings().weight.device
        self.dtype = model.get_input_embeddings().weight.dtype
        self.q = None if q is None else torch.as_tensor(q / rms, dtype=torch.float32, device=self.device)
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
        if 'input' in self.config:
            # The full semantic state belongs in the key when either textual ID
            # is visible. Avoid assuming latent-only prompts for new conditions.
            key = (item['u'], item['g'], item['task'], item['template'], text_control)
            if key not in self.prompt_cache:
                prompt = build_prompt(item, item['task'], self.config['input'], self.config, self.adjacency)
                ids = self.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
                for role, slot in zip(('current', 'goal'), self.slot_ids):
                    if ids.count(slot) != int(self.config['input'][role + '_vector']):
                        raise ValueError('Prompt state slots do not match input config')
                self.prompt_cache[key] = ids
            return self.prompt_cache[key]
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
                from .training import answer_tokens
                answer = answer_tokens(self.config, self.tokenizer, item, self.end_id,
                                       None if answer_override is None else answer_override[index])
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
        embeddings = inject_vectors(embeddings, ids, dict(zip(('u', 'g'), self.slot_ids)),
                                    items, self.q, adapter, self.dtype)
        positions = attention.cumsum(dim=-1) - 1
        positions.masked_fill_(attention == 0, 0)
        return dict(inputs_embeds=embeddings, attention_mask=attention,
                    position_ids=positions, labels=labels, input_ids=ids)

    def losses(self, items, adapter):
        from .training import losses
        return losses(self, items, adapter)

    def generate(self, items, adapter, *, prefix_allowed_tokens_fn=None):
        from .inference import generate
        return generate(self, items, adapter, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)

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
        item = dict(u=0, g=1, task='report_current', template='canonical')
        prompt = self.prompt_ids(item, False)
        self.prefix_length = prompt.index(self.slot_ids[0])
        self.prefix_ids = prompt[:self.prefix_length]
        with torch.no_grad():
            output = self.model.model(
                input_ids=torch.tensor([self.prefix_ids],device=self.device), use_cache=True)
        self.prefix_kv = [(layer[0].detach(),layer[1].detach()) for layer in output.past_key_values]

    def losses(self, items, adapter):
        from .training import cached_losses
        return cached_losses(self, items, adapter)

class AugmentedInterface(StateInterface):
    def __init__(self, config, spec, *args):
        self.spec = spec
        super().__init__(config, *args)

    def prompt_ids(self, item, text_control):
        # Every prompt includes explicit IDs, even when an adapter is present.
        key = (item['u'], item['g'], item['task'], item['template'], text_control)
        if key not in self.prompt_cache:
            ids = self.tokenizer.apply_chat_template(
                augmentation_messages(self.spec, self.adjacency, item, not text_control,
                                      self.config['assets']['node_count']),
                tokenize=True, add_generation_prompt=True, return_dict=False)
            if not text_control:
                assert all(ids.count(slot) == 1 for slot in self.slot_ids)
            self.prompt_cache[key] = ids
        return self.prompt_cache[key]

class DecisionInterface(StateInterface):
    def __init__(self,model,tokenizer,q,rms,adjacency,permutation):
        self.model=model.eval().requires_grad_(False);self.tokenizer=tokenizer
        self.device=model.device;self.dtype=model.get_input_embeddings().weight.dtype
        self.q=None if q is None else torch.tensor(q/rms,dtype=torch.float32,device=self.device)
        self.adjacency=adjacency;self.permutation=permutation
        tokenizer.add_tokens([f'<|diag_{r}|>' for r in ('a','b','g')],special_tokens=True)
        self.slots={r:tokenizer.convert_tokens_to_ids(f'<|diag_{r}|>') for r in ('a','b','g')}
        self.eos=tokenizer.eos_token_id
        self.pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos
    def batch(self, items, condition, adapter, reasons=None):
        sequences=[self.tokenizer.apply_chat_template(decision_messages(x,condition,self.adjacency,None if reasons is None else reasons[i]),
            tokenize=True,add_generation_prompt=True,return_dict=False) for i,x in enumerate(items)]
        width=max(map(len,sequences))
        ids=torch.full((len(items),width),self.pad,dtype=torch.long,device=self.device)
        mask=torch.zeros_like(ids)
        for i,seq in enumerate(sequences):
            ids[i,-len(seq):]=torch.tensor(seq,device=self.device)
            mask[i,-len(seq):]=1
        safe=ids.clone()
        for slot in self.slots.values():
            safe[safe==slot]=self.pad
        embeds=self.model.get_input_embeddings()(safe)
        embeds=inject_vectors(embeds, ids, self.slots, items, self.q, adapter, self.dtype,
                              self.permutation if condition.endswith('_mismatch') else None)
        return embeds,mask,width
    def generate(self,items,condition,adapter,reasons=None,analysis=False,reason_tokens=512):
        from .inference import generate_choices
        return generate_choices(self, items, condition, adapter, reasons=reasons, analysis=analysis, reason_tokens=reason_tokens)

CachedReportInterface = CachedStateInterface
