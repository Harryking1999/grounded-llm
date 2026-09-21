"""Authority for generation and output grammars; never chooses graph actions."""
import torch
from transformers import GenerationConfig

@torch.no_grad()
def generate_ids(model, embeddings, attention, generation_config, grammar=None, stopping_criteria=None):
    """The single LM generation call shared by node and choice protocols."""
    if embeddings.shape[1] + generation_config.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError('Prompt plus generation budget exceeds model context')
    return model.generate(inputs_embeds=embeddings, attention_mask=attention,
                          generation_config=generation_config, prefix_allowed_tokens_fn=grammar,
                          stopping_criteria=stopping_criteria)

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

class ChoiceGrammar:
    def __init__(self,tokenizer,options):
        self.eos=tokenizer.eos_token_id
        self.pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos
        self.tables=[];self.max_tokens=0
        for choices in options:
            table={}
            for choice in choices:
                seq=tokenizer.encode(choice,add_special_tokens=False)+[self.eos]
                self.max_tokens=max(self.max_tokens,len(seq))
                for i,token in enumerate(seq):table.setdefault(tuple(seq[:i]),set()).add(token)
            self.tables.append(table)
    def __call__(self,batch_id,ids):
        prefix=tuple(ids.tolist())
        return [self.pad] if self.eos in prefix else sorted(self.tables[batch_id][prefix])

def pending_items(items, rows):
    """Resume one condition from complete saved records, keeping original order."""
    done = {(r['u'],r['g']) for r in rows}
    return [r for r in items if (r['u'],r['g']) not in done]

@torch.no_grad()
def generate(self, items, adapter, *, prefix_allowed_tokens_fn=None):
    batch = self.batch(items, adapter)
    # With inputs_embeds only, HF returns just the newly generated IDs.
    generated = generate_ids(self.model, batch['inputs_embeds'], batch['attention_mask'],
                             self.generation, prefix_allowed_tokens_fn)
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
def generate_choices(self,items,condition,adapter,reasons=None,analysis=False,reason_tokens=512):
    embeds,mask,width=self.batch(items,condition,adapter,reasons)
    grammar=None if analysis else ChoiceGrammar(self.tokenizer,[x['options'] for x in items])
    budget=reason_tokens if analysis else grammar.max_tokens
    cfg=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=budget,eos_token_id=self.eos,pad_token_id=self.pad,use_cache=True)
    generated=generate_ids(self.model,embeds,mask,cfg,grammar)
    out=[]
    for row in generated.tolist():
        ended=self.eos in row;seq=row[:row.index(self.eos)] if ended else row
        out.append(dict(output=self.tokenizer.decode(seq,skip_special_tokens=True,clean_up_tokenization_spaces=False),tokens=len(seq),ended=ended))
    return out
