"""Restoration task evaluation using the existing SFT model and result schema."""
import json
import numpy as np
import torch
from transformers import AutoTokenizer, StoppingCriteriaList
from grounded_llm.core.files import atomic_json
from grounded_llm.core.reader.batching import timeline_maps, to_device
from grounded_llm.core.sft.modeling import build
from grounded_llm.tasks.restoration_sft import environment, load_map, make_step, prompt, score
from experiments.flamingo_map_reader.src.counterfactual import reverse_candidate_q
from .metrics import summarize


def evaluate(args, config):
    from .evaluate import Boundary
    task = config["task"]
    env = environment(task, args.data / "oracle.npy")
    qmap = load_map(task, env, args.q_checkpoint)
    records = json.loads((args.data / "evaluation.json").read_text())
    if len(records) != config["evaluation"]["tasks"]:
        raise ValueError("Evaluation cohort mismatch")
    if args.mode == "probe":
        records = [records[i] for i in json.loads((args.data / "probe_indices.json").read_text())]
    records = records[args.start:args.stop]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = build(config, args.baseline, args.model_path, False, args.checkpoint).cuda()
    args.out.mkdir(parents=True, exist_ok=True)

    def generate(messages, steps, map_ids):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(ids) + config["evaluation"]["max_new_tokens"] > config["maximum_sequence_tokens"]:
            return "<context_overflow/>"
        inputs = torch.tensor([ids], device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=config["evaluation"]["max_new_tokens"], do_sample=False,
                use_cache=True, pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([Boundary(tokenizer, len(ids))]),
                map_batch=to_device(timeline_maps(steps, map_ids), torch.device("cuda")))
        return tokenizer.decode(output[0, len(ids):], skip_special_tokens=True).strip()

    cases = []
    with (args.out / "cases.jsonl").open("w") as output:
        for record in records:
            state, goal = record["start"], record["goal"]
            executed, messages, steps, previous_ids, turns = [], [], [], [], []
            arrived, stopped = state == goal, False
            limit = (config["evaluation"]["action_budget"] if args.mode == "RO"
                     else len(record["reference_actions"]))
            for index in range(limit + 1):
                step = make_step(env, qmap, state, goal)
                if args.mode != "RO":
                    messages, steps, previous_ids = [], [], []
                update = prompt(task, env, step, executed, record["start"], not messages)
                messages.append(dict(role="user", content=update))
                steps.append(step)
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                map_ids = previous_ids + [len(steps) - 1] * (len(ids) - len(previous_ids))
                text = generate(messages, steps, map_ids)
                row = score(text, step, env)
                turns.append(row)
                if args.mode == "probe" and not step.done:
                    altered, _ = reverse_candidate_q(step)
                    other = generate(messages, [altered], [0] * len(ids))
                    row["reverse"] = score(other, step, env)
                    row["changed_action"] = row["candidate"] != row["reverse"]["candidate"]
                if args.mode != "RO":
                    if index == len(record["reference_actions"]):
                        break
                    action = record["reference_actions"][index]
                    executed.append(action)
                    state = env.step(state, action)
                    continue
                if step.done:
                    arrived, stopped = True, row["candidate"] is None and row["error"] is None
                    break
                if row["error"] or row["candidate"] is None or index == limit:
                    break
                action, state = step.execute(row["candidate"])
                executed.append(action)
                arrived |= state == goal
                messages.append(dict(role="assistant", content=text))
                complete = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
                previous_ids = map_ids + [len(steps) - 1] * (len(complete) - len(map_ids))
            case = dict(trajectory_id=record["trajectory_id"], arrived=arrived,
                        arrived_and_stopped=arrived and stopped, turns=turns)
            output.write(json.dumps(case) + "\n")
            output.flush()
            cases.append(case)
            print(json.dumps(dict(done=len(cases), total=len(records))), flush=True)
    atomic_json(args.out / "summary.json", dict(mode=args.mode, baseline=args.baseline,
        checkpoint=json.loads((args.checkpoint / "evaluation_ready.json").read_text()),
        metrics=summarize(cases)))
