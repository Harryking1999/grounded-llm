"""Matched first-step generation and input interventions; never optimize weights."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random
import re


def select_records(records, count, seed, distinct_boards=False):
    eligible = [r for r in records if int(r['start']) != int(r['goal'])]
    random.Random(seed).shuffle(eligible)
    selected, seen = [], set()
    for row in eligible:
        key = row['board_row'] if distinct_boards else row['trajectory_id']
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) == count:
            return selected
    raise ValueError('Insufficient distinct nonterminal samples')


def first_rank_candidate(text):
    line = next((s for s in text.splitlines() if s.startswith('Map-distance ranking')), '')
    match = re.search(r'closest to farthest:\s*.*?([1-9][0-9]*)', line)
    return int(match.group(1)) if match else None


def reference_candidate(action_keys, reference_key):
    """JSON stores reference tuples as lists; compare physical moves consistently."""
    return [tuple(key) for key in action_keys].index(tuple(reference_key))+1


def donor_records(records):
    """Use legal intermediate training states, rather than changing the test cohort."""
    return [dict(start=str(state), goal=str(goal), trajectory_id=f"{row['trajectory_id']}:turn-{i}")
            for row in records for i, (state, goal) in enumerate(row['pairs'])
            if int(state) != int(goal)]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['split'], row['condition'])].append(row)
    result = {}
    for (split, condition), items in groups.items():
        n = len(items)
        answers = Counter(x['answer'].strip() for x in items)
        entry = dict(tasks=n, unique_answers=len(answers),
                     duplicate_fraction=1-len(answers)/n,
                     dominant_answer_fraction=max(answers.values())/n)
        for key in ['legal', 'optimal', 'reference_action_match', 'rank_map_optimal']:
            entry[key] = sum(bool(x[key]) for x in items)/n
        paired = [x for x in items if 'action_changed' in x]
        if paired:
            entry['action_change_rate'] = sum(x['action_changed'] for x in paired)/len(paired)
            entry['answer_change_rate'] = sum(x['answer_changed'] for x in paired)/len(paired)
            entry['mean_reference_action_logp_delta'] = sum(x['reference_action_logp_delta'] for x in paired)/len(paired)
            entry['mean_absolute_reference_action_logp_delta'] = sum(abs(x['reference_action_logp_delta']) for x in paired)/len(paired)
        result[f'{split}/{condition}'] = entry
    return result


def main():
    import torch
    from transformers import AutoTokenizer, StoppingCriteriaList
    from grounded_llm.tasks.blocks.oracle import DistanceOracle
    from grounded_llm.tasks.blocks.qmap import FrozenBoardMap
    from experiments.flamingo_map_reader.src.graph import timeline_maps
    from grounded_llm.core.reader.batching import to_device
    from grounded_llm.tasks.blocks.protocol import legal_actions, action_key
    from grounded_llm.tasks.blocks.sft import prompt, sorted_step, atomic_json
    from .evaluate import score, Boundary
    from .modeling import build
    from .prepare import CachedMap

    parser = argparse.ArgumentParser()
    for key in ['config', 'study-config', 'training-data', 'evaluation-data', 'training-root', 'model-path', 'q-checkpoint', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--split', choices=['all', 'train', 'test'], default='all')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    policy = json.loads(args.config.read_text())
    if args.epoch not in policy['epochs']:
        raise ValueError('Epoch is outside the committed audit contract')
    study = json.loads(args.study_config.read_text())
    base = policy['baseline']
    step_number = args.epoch*study['data']['samples_per_epoch']//study['training']['global_batch']
    checkpoint = args.training_root/f'checkpoint-{step_number}'
    metadata = json.loads((checkpoint/'evaluation_ready.json').read_text())
    if metadata != dict(step=step_number, epoch=float(args.epoch), baseline=base):
        raise ValueError('Checkpoint identity mismatch')
    train = json.loads((args.training_data/'dataset.json').read_text())['rows']
    test = json.loads((args.evaluation_data/'evaluation.json').read_text())
    selected = dict(train=select_records(train, policy['samples_per_split'], policy['seed'], True),
                    test=select_records(test, policy['samples_per_split'], policy['seed']+1))
    # Donors preserve slot count and addresses, while carrying a different physical task.
    donors = defaultdict(list)
    donor_pool = donor_records(train)
    random.Random(policy['seed']+2).shuffle(donor_pool)
    counts = {len(legal_actions(int(x['start']))) for rows in selected.values() for x in rows}
    for row in donor_pool:
        if int(row['start']) == int(row['goal']):
            continue
        count = len(legal_actions(int(row['start'])))
        if count in counts and len(donors[count]) < 2 and all(x['start'] != row['start'] for x in donors[count]):
            donors[count].append(row)
        if all(len(donors[c]) == 2 for c in counts):
            break
    unmatched = [row['trajectory_id'] for records in selected.values() for row in records
                 if not any(d['start'] != row['start']
                            for d in donors[len(legal_actions(int(row['start'])))])]
    if unmatched:
        raise ValueError('No distinct-state matched donor for '+str(unmatched))
    args.out.mkdir(parents=True, exist_ok=True)
    cohort = dict(seed=policy['seed'], splits=selected, donors=dict(donors))
    cohort_path = args.out/'cohort.json'
    if cohort_path.exists() and json.loads(cohort_path.read_text()) != json.loads(json.dumps(cohort)):
        raise ValueError('Fixed cohort changed')
    atomic_json(cohort_path, cohort)
    if args.prepare_only:
        return
    destination = args.out/f'epoch-{args.epoch}'
    if args.split != 'all':
        destination = destination/args.split
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'summary.json').exists():
        return
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = build(study, base, args.model_path, False, checkpoint).cuda()
    qmap = CachedMap(FrozenBoardMap.load(args.q_checkpoint))
    oracle = DistanceOracle(cache_limit=2000000, seconds=86400)

    def prefix(row):
        text = prompt(int(row['start']), int(row['goal']), [], True, int(row['start']), True,
                      study['baselines'][base]['response_order'])
        return tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=True,
                                             add_generation_prompt=True)

    def cached_prefix(row):
        item = torch.load(args.training_data/'trajectories'/row['prepared_file'],
                          map_location='cpu', weights_only=False)['map']['encoded']
        first = next(i for i, label in enumerate(item.labels) if label != -100)
        return item.input_ids[:first]

    def generate(ids, map_step):
        timeline = to_device(timeline_maps([map_step], [0]*len(ids)), torch.device('cuda'))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = model.generate(input_ids=torch.tensor([ids], device='cuda'),
                attention_mask=torch.ones((1, len(ids)), device='cuda', dtype=torch.long),
                map_batch=timeline, max_new_tokens=policy['max_new_tokens'], do_sample=False,
                use_cache=True, pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([Boundary(tokenizer, len(ids))]))
        return tokenizer.decode(output[0, len(ids):], skip_special_tokens=True).strip()

    def action_logp(ids, map_step, target):
        # A separate, deliberately forced readout. It contains no answer-bearing reasoning.
        context = ids+tokenizer.encode('<action>', add_special_tokens=False)
        digits = tokenizer.encode(str(target), add_special_tokens=False)
        tokens = context+digits
        timeline = to_device(timeline_maps([map_step], [0]*len(tokens)), torch.device('cuda'))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), model.conditioned(timeline):
            hidden = model.base_model.model(input_ids=torch.tensor([tokens], device='cuda'),
                attention_mask=torch.ones((1, len(tokens)), device='cuda', dtype=torch.long),
                use_cache=False).last_hidden_state[:, len(context)-1:len(tokens)-1]
            logits = model.base_model.lm_head(hidden).float().log_softmax(-1)
            return float(logits[0, torch.arange(len(digits), device='cuda'), torch.tensor(digits, device='cuda')].sum())

    rows = []
    with (destination/'cases.jsonl').open('w') as output:
        for split, records in selected.items():
            if args.split != 'all' and split != args.split:
                continue
            for index, record in enumerate(records):
                original = sorted_step(qmap, int(record['start']), int(record['goal']))
                original_ids = cached_prefix(record) if split == 'train' else prefix(record)
                cache_match = original_ids == prefix(record) if split == 'train' else None
                count = len(original.candidate_actions)
                donor = next(x for x in donors[count] if x['start'] != record['start'])
                other = sorted_step(qmap, int(donor['start']), int(donor['goal']))
                reference = (record['route_ids'][0] if split == 'train' else
                    reference_candidate([action_key(action) for action in legal_actions(int(record['start']))],
                                        record['reference_actions'][0]))
                conditions = policy['conditions'] if args.epoch == policy['intervention_epoch'] else ['natural']
                natural = None
                for condition in conditions:
                    text_record = donor if condition == 'text_swap' else record
                    map_step = other if condition == 'map_swap' else original
                    if condition == 'map_zero_vectors':
                        map_step = replace(original, map_batch=replace(original.map_batch,
                                                vectors=torch.zeros_like(original.map_batch.vectors)))
                    ids = prefix(donor) if condition == 'text_swap' else original_ids
                    answer = generate(ids, map_step)
                    scored = score(answer, original, oracle, True)
                    rank = first_rank_candidate(answer)
                    row = dict(split=split, trajectory_id=record['trajectory_id'], epoch=args.epoch,
                        condition=condition, donor_id=donor['trajectory_id'], answer=answer,
                        action=scored['candidate'], error=scored['error'],
                        legal=scored['candidate'] is not None and scored['error'] is None,
                        optimal=scored['optimal'], reference_action_match=scored['candidate'] == reference,
                        first_rank_candidate=rank, rank_map_optimal=rank in original.map_minimal_candidates)
                    row['regenerated_prefix_matches_training_cache'] = cache_match
                    if args.epoch == policy['intervention_epoch']:
                        row['reference_action_logp'] = action_logp(ids, map_step, reference)
                    if natural is None:
                        natural = row
                    else:
                        row.update(action_changed=(row['action'],row['error']) != (natural['action'],natural['error']),
                            answer_changed=answer != natural['answer'],
                            reference_action_logp_delta=row['reference_action_logp']-natural['reference_action_logp'])
                    rows.append(row)
                    output.write(json.dumps(row)+'\n')
                    output.flush()
                atomic_json(destination/'progress.json', dict(split=split, completed=index+1,
                            total=len(records), cases=len(rows)))
    atomic_json(destination/'summary.json', dict(epoch=args.epoch, checkpoint=metadata,
                metrics=summarize(rows), contract=policy))


if __name__ == '__main__':
    main()
