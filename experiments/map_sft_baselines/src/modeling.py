"""Existing reader architecture, trainable-scope switch and uniform token CE."""
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM
from experiments.flamingo_map_reader.src.train import build_reader

class TokenLM(nn.Module):
    def __init__(self,base,chunk):
        super().__init__();self.base=base;self.chunk=chunk
    def forward(self,input_ids,attention_mask,labels):
        hidden=self.base.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False).last_hidden_state[:,:-1]
        target=labels[:,1:];selected=target.ne(-100);features=hidden[selected];target=target[selected]
        def ce(x,y):return nn.functional.cross_entropy(self.base.lm_head(x).float(),y,reduction='sum')
        loss=sum(checkpoint(ce,x,y,use_reentrant=False) if self.training else ce(x,y)
                 for x,y in zip(features.split(self.chunk),target.split(self.chunk)))
        return loss/len(target)
    def generate(self,**kwargs):return self.base.generate(**kwargs)

def build(config,baseline,model_path,training=True,checkpoint_path=None):
    spec=config['baselines'][baseline]
    base=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.float32 if training else torch.bfloat16,attn_implementation='sdpa')
    base.config.use_cache=not training
    if spec['interface']:
        model=build_reader(base,config,sequence_mean_loss=False,cache_generation=not training)
        if spec['full_llm']:base.requires_grad_(True)
    else:
        model=TokenLM(base,config['training']['loss_chunk_tokens'])
        if training:base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if checkpoint_path:
        saved=torch.load(Path(checkpoint_path)/'weights.pt',map_location='cpu',weights_only=True)
        if saved['baseline']!=baseline:raise ValueError('Checkpoint baseline mismatch')
        if spec['interface'] and not spec['full_llm']:model.load_adapter_state_dict(saved['weights'])
        else:model.load_state_dict(saved['weights'])
    model.train(training)
    return model

def infer_weights(model,config,baseline):
    spec=config['baselines'][baseline]
    weights=model.adapter_state_dict() if spec['interface'] and not spec['full_llm'] else model.state_dict()
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().to(device='cpu',dtype=torch.bfloat16) if value.is_floating_point() else value.cpu()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        return [cpu(v) for v in value]
    return dict(baseline=baseline,weights=cpu(weights))
