"""Shared interpretation of training and evaluation budgets."""



from grounded_llm.core.sft.schedule import steps_per_epoch, training_steps

def checkpoint_priority(step, config):
    """Milestones, then full epochs, then other checkpoints; newest in each."""
    preferred = config.get('evaluation', {}).get('priority_steps', [])
    if step in preferred:
        return (-1, preferred.index(step))
    interval = config.get('evaluation', {}).get('priority_step_interval', 0)
    if interval and step > 0 and step % interval == 0:
        return (0, -step)
    if config.get('evaluation', {}).get('prioritize_full_epochs', False):
        return (1 if step > 0 and step % steps_per_epoch(config) == 0 else 2, -step)
    return (1, -step)

def checkpoint_steps(config):
    interval = config['training']['save_steps']
    total = training_steps(config)
    if interval <= 0 or total % interval:
        raise ValueError('Checkpoint interval must divide the training horizon')
    return range(interval, total + 1, interval)




def cohort_size(config, mode):
    key = 'probe_tasks' if mode == 'probe' else 'tasks'
    return config['evaluation'][key]


def evaluation_modes(config, baseline):
    modes = list(config['evaluation']['modes'])
    if config['baselines'][baseline]['real_map'] and config['evaluation'].get('probe_tasks', 0) > 0:
        modes.append('probe')
    return modes


def plateau_status(curve, policy, interval):
    """Only contiguous complete checkpoints count toward the patience window."""
    complete = []
    for row in sorted(curve, key=lambda item: item['step']):
        if row['step'] != (len(complete) + 1) * interval:
            break
        complete.append(row)
    patience = policy['patience_points']
    if len(complete) <= patience or complete[-1]['epoch'] < policy['minimum_epochs']:
        return None
    previous, recent = complete[:-patience], complete[-patience:]
    best = max(row['metrics']['RO']['arrival'] for row in previous)
    if max(row['metrics']['RO']['arrival'] for row in recent) >= best + policy['minimum_gain_tasks']:
        return None
    return dict(status='plateau_flag', through_epoch=complete[-1]['epoch'],
                previous_best=best, mode=policy['mode'])
