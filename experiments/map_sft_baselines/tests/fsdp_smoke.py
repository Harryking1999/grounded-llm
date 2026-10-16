"""Four-GPU numerical/update/export check with a tiny Qwen2; no research scores."""
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from transformers import Qwen2Config, Qwen2ForCausalLM
from experiments.map_sft_baselines.src.modeling import TokenLM
from experiments.map_sft_baselines.src.train import shard_model, save_sharded


def main():
    rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);dist.init_process_group('nccl')
    torch.manual_seed(17)
    c=Qwen2Config(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
                 num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
                 tie_word_embeddings=False,use_cache=False)
    base=Qwen2ForCausalLM(c);base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model=shard_model(TokenLM(base,8),local)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,fused=True)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    torch.manual_seed(100+rank)
    ids=torch.randint(0,128,(1,32),device=local)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(ids,torch.ones_like(ids),ids)
    loss.backward();norm=model.clip_grad_norm_(1.)
    assert torch.isfinite(loss) and torch.isfinite(norm)
    opt.step();scheduler.step()
    out=Path(sys.argv[1]);save_sharded(model,opt,scheduler,out,dict(training={'keep_resume_checkpoints':1}), 'smoke',1,1,rank)
    if rank==0:
        saved=torch.load(out/'checkpoint-1/weights.pt',map_location='cpu',weights_only=True)
        restored=TokenLM(Qwen2ForCausalLM(c),8)
        restored.load_state_dict(saved['weights'],strict=True)
        assert all(torch.isfinite(p).all() for p in restored.parameters())
        resume=torch.load(out/'checkpoint-1/resume.pt',map_location='cpu',weights_only=False)
        assert len(resume['optimizer']['state'])>0
        print('FSDP_UPDATE_AND_EXPORT_OK',float(loss),float(norm),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
