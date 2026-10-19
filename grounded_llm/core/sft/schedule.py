"""Training budgets, learning-rate schedule and resume compatibility."""
import math

def scheduler_factor(step, total, warmup, floor):
    if step < warmup:
        return step / max(1, warmup)
    progress = min(1., (step - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


def steps_per_epoch(config):
    samples = config['data']['samples_per_epoch']
    batch = config['training']['global_batch']
    if samples <= 0 or batch <= 0 or samples % batch:
        raise ValueError('Samples must divide into complete positive global batches')
    return samples // batch


def training_steps(config):
    horizon = steps_per_epoch(config) * config['training']['epochs']
    total = config['training'].get('maximum_updates', horizon)
    if not isinstance(total, int) or not 0 < total <= horizon:
        raise ValueError('Update limit must be positive and within the epoch horizon')
    return total


def learning_rate_factor(step,total,training):
    total=training.get('scheduler_total_steps',total)
    if total <= 0:raise ValueError('Scheduler horizon must be positive')
    warmup=training.get('warmup_steps',round(total*training['warmup_fraction']))
    schedule=training.get('scheduler','cosine')
    if schedule=='constant_with_warmup':return min(1.,step/max(1,warmup))
    if schedule!='cosine':raise ValueError('Unsupported learning-rate scheduler')
    return scheduler_factor(step,total,warmup,training['cosine_minimum_ratio'])


def validate_resume_contract(saved,config,baseline,allow_horizon_extension=False):
    if allow_horizon_extension:
        previous=saved['config']
        old_training,new_training=previous['training'],config['training']
        old_total,new_total=training_steps(previous),training_steps(config)
        # Only an explicit longer horizon is permitted; the LR prefix must be identical.
        if (new_total<=old_total or
                new_training.get('scheduler_total_steps',0)<new_total or
                old_training.get('scheduler_total_steps')!=new_training.get('scheduler_total_steps')):
            raise ValueError('Extension requires a longer budget and an unchanged explicit scheduler horizon')
        old_recipe={k:v for k,v in old_training.items() if k not in ('epochs','maximum_updates')}
        new_recipe={k:v for k,v in new_training.items() if k not in ('epochs','maximum_updates')}
        if old_recipe!=new_recipe:
            raise ValueError('Extension may only change epochs and maximum_updates')
        from copy import deepcopy
        saved=deepcopy(saved)
        saved['config']['training']=dict(new_training)
    recipe=('model','seed','data','map','training','maximum_sequence_tokens')
    if (saved['baseline']!=baseline or saved['config']['baselines'][baseline]!=config['baselines'][baseline]
            or any(saved['config'][key]!=config[key] for key in recipe)):
        raise ValueError('Resume training recipe mismatch')
