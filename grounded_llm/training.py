"""Authority for answer loss, optimizer steps and adapter checkpoints."""
import torch
import time
import numpy as np
from transformers import DynamicCache
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from .artifacts import chunks, append_json, write_json

def answer_tokens(config, tokenizer, item, end_id, override=None):
    """Construct only authorized teacher-forcing targets, including native EOS."""
    if item['task'] not in config['training']['tasks']:
        raise ValueError('Task is not authorized for training')
    target = item['target'] if item['task'] == 'action' else item['u'] if item['task'] == 'report_current' else item['g']
    return tokenizer.encode(str(target) if override is None else override, add_special_tokens=False) + [end_id]

def load_adapter(path, adapter, device):
    adapter.load_state_dict(load_file(str(path), device=str(device)))
    return adapter

def joint_loss_plateau(history, spec):
    """Operational plateau of both losses, never a claim of successful learning."""
    checks = [row for row in history if 'validation_loss' in row]
    if not spec or not checks or checks[-1]['epoch'] < spec['minimum_epochs'] or len(checks) < spec['checks']:
        return False
    window = checks[-spec['checks']:]
    for key in ('loss', 'validation_loss'):
        first = window[0][key]
        improvement = (first - min(row[key] for row in window[1:])) / max(abs(first), 1e-12)
        if improvement >= spec['relative_min_improvement']:
            return False
    first_accuracy = window[0]['validation_correct'] / window[0]['validation_total']
    return max(row['validation_correct'] / row['validation_total'] for row in window[1:]) <= first_accuracy


def train(interface, adapter, items, validation, output, validate, resume=None):
    """One training loop for report, action and matched replay conditions.

    validate is supplied by the harness. It returns (correct, total), keeping
    the task judge outside loss/optimizer/checkpoint code.
    """
    spec = interface.config['training']
    microbatch = interface.config.get('runtime', {}).get('microbatch_size', spec['initial_microbatch_size'])
    optimizer = optimizer_for(interface.config, adapter)
    rng = np.random.default_rng(spec['shuffle_seed'])
    best, selected_epoch, selected_state = None, None, None
    start_epoch, history = 0, []
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=True)
        previous_spec = state['training_spec']
        if ({k: v for k, v in previous_spec.items() if k != 'epochs'} !=
                {k: v for k, v in spec.items() if k != 'epochs'} or state['samples'] != len(items)):
            raise ValueError('Resume must retain optimizer, supervision and dataset size')
        adapter.load_state_dict(state['adapter'])
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['shuffle_state']
        torch.set_rng_state(state['torch_rng_state'])
        start_epoch, history = state['completed_epoch'], state['history']
        best, selected_epoch, selected_state = state['best'], state['selected_epoch'], state['selected_adapter']
        if start_epoch >= spec['epochs']:
            raise ValueError('No unfinished training epochs in requested budget')
        if selected_state is not None:
            save_file(selected_state, str(output / 'selected.safetensors'))
    save_adapter(output / 'initial.safetensors', adapter)
    stop_reason = 'budget_exhausted'
    for epoch in range(start_epoch + 1, spec['epochs'] + 1):
        if hasattr(items, 'set_epoch'):
            items.set_epoch(epoch)
        ordered = [items[int(i)] for i in rng.permutation(len(items))]
        total_loss = 0.0
        started = time.perf_counter()
        for step, batch in enumerate(chunks(ordered, spec['report_batch_size']), 1):
            loss, grad = update(interface, adapter, optimizer, batch, microbatch)
            total_loss += loss * len(batch)
            append_json(output / 'train.jsonl', dict(epoch=epoch, step=step, samples=len(batch), loss=loss, gradient_norm=grad))
        row = dict(epoch=epoch, loss=total_loss / len(items), seconds=time.perf_counter() - started)
        if spec['selection'] == 'validation' and (epoch % spec.get('validation_interval', 1) == 0 or epoch == spec['epochs'] or (epoch == 1 and spec.get('validate_first_epoch'))):
            if spec.get('validation_loss_from_callback'):
                correct, count, val_loss = validate(interface, adapter, validation, epoch)
            else:
                val_loss = mean_loss(interface, adapter, validation, microbatch)
                correct, count = validate(interface, adapter, validation, epoch)
            rank = (-correct / count, val_loss, epoch)
            if best is None or rank < best:
                best, selected_epoch = rank, epoch
                save_adapter(output / 'selected.safetensors', adapter)
                selected_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
            row.update(validation_loss=val_loss, validation_correct=correct, validation_total=count, selected_epoch=selected_epoch)
        append_json(output / 'validation.jsonl', row)
        history.append(row)
        plateau = joint_loss_plateau(history, spec.get('convergence'))
        if spec.get('save_training_state') and ('validation_loss' in row or epoch == spec['epochs']):
            # One replaceable checkpoint, written atomically; preserves optimizer and shuffle on restart.
            temporary = output / 'latest.pt.tmp'
            torch.save(dict(adapter=adapter.state_dict(), optimizer=optimizer.state_dict(),
                            shuffle_state=rng.bit_generator.state, torch_rng_state=torch.get_rng_state(),
                            completed_epoch=epoch, history=history, best=best, selected_epoch=selected_epoch,
                            selected_adapter=selected_state, training_spec=spec, samples=len(items)), temporary)
            temporary.replace(output / 'latest.pt')
        if 'validation_correct' in row:
            print(row, flush=True)
        if plateau:
            stop_reason = 'joint_loss_plateau'
            break
    save_adapter(output / 'final.safetensors', adapter)
    if spec['selection'] == 'final':
        selected_epoch = epoch
        save_adapter(output / 'selected.safetensors', adapter)
    result = dict(selected_epoch=selected_epoch, final_epoch=epoch, requested_epochs=spec['epochs'],
                  stop_reason=stop_reason, resumed_from_epoch=start_epoch,
                  selected_validation_accuracy=-best[0] if best else None,
                  selected_validation_loss=best[1] if best else None,
                  samples=len(items), sample_presentations=len(items) * epoch)
    write_json(output / 'training_summary.json', result)
    return result

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

def optimizer_for(config, adapter):
    train = config['training']
    return torch.optim.AdamW(adapter.parameters(), lr=train['learning_rate'],
                            betas=tuple(train['betas']), eps=train['epsilon'],
                            weight_decay=train['weight_decay'])

def update(interface, adapter, optimizer, items, microbatch):
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for micro in chunks(items, microbatch):
        losses = interface.losses(micro, adapter)
        loss = losses.sum() / len(items)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        total += float(loss.detach())
    grad = torch.nn.utils.clip_grad_norm_(adapter.parameters(), interface.config['training']['gradient_clip_norm'],
                                         error_if_nonfinite=True)
    if not torch.isfinite(grad) or grad <= 0:
        raise FloatingPointError('Adapter gradient must be finite and nonzero')
    if any(p.grad is not None for model in getattr(interface, 'frozen_models', [interface.model]) for p in model.parameters()):
        raise RuntimeError('Frozen LM received parameter gradients')
    optimizer.step()
    return total, float(grad)

@torch.no_grad()
def mean_loss(interface, adapter, items, microbatch):
    return sum(float(interface.losses(micro, adapter).sum()) for micro in chunks(items, microbatch)) / len(items)

def save_adapter(path, adapter):
    save_file({k: v.detach().cpu().contiguous() for k, v in adapter.state_dict().items()}, str(path))

def checkpoint_for_condition(condition, spec, base, continuation=None):
    """Resolve one adapter checkpoint from the formal condition, never from a run name guess."""
    if condition['adapter'] is None:
        return None
    source = condition.get('checkpoint_source', 'base')
    if source == 'base':
        root = base
    elif source == 'continuation' and continuation is not None:
        root = continuation
    else:
        raise ValueError(f'Unavailable checkpoint source: {source}')
    name = condition.get('checkpoint_condition', condition.get('base_condition'))
    if not name:
        raise ValueError('Adapter condition needs a checkpoint condition')
    checkpoint = condition.get('checkpoint', spec['checkpoint'])
    path = root / name / (checkpoint + '.safetensors')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

def losses(self, items, adapter):
    from .parallel_readout import readout
    batch = self.batch(items, adapter, supervised=True)
    return readout(self, batch, supervised=True)

def cached_losses(self, items, adapter):
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
