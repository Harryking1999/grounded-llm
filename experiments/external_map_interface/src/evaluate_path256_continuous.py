"""One-shot routes and controller-enforced node-level map interventions."""

import argparse
import json
from pathlib import Path
import re
import time
import urllib.request
import uuid

from .q_map import GraphQMap
from .transitions import environment_from_suite


PROTOCOL = "controller_node_boundary"
ENVIRONMENT_MARKER = "[Environment update]"
OPENING = re.compile(r'<action>|\{\s*"path"\s*:\s*\[|\[Environment update\]')
INTEGER = re.compile(r"\s*(0|[1-9][0-9]*)\s*([,\]])")
FINAL_PATTERN = re.compile(r'(\{[^{}]*\})\s*$', re.DOTALL)
FINAL_FENCED_PATTERN = re.compile(
    r'(?:^|\n)```(?:json)?[ \t]*\r?\n\s*(\{[^{}]*\})\s*\r?\n```[ \t]*$',
    re.DOTALL | re.IGNORECASE,
)


def common_prompt(case):
    neighbors = "\n".join(
        f"{node}: {', '.join(map(str, case['neighbors'][str(node)]))}"
        for node in case["node_order"]
    )
    return (
        f"Find a valid path from node {case['start']} to node {case['goal']} in this undirected graph, as short as you can.\n"
        "Use the listed edges and visit each node at most once.\n\n"
        f"Neighbors:\n{neighbors}"
    )


def trial_prompt(case, condition):
    prompt = common_prompt(case)
    if not condition["map_updates"]:
        output_format = (
            'Return the complete route as one final JSON object: {"path": [<start>, ..., <goal>]}.\n'
            'You may reason before the final JSON. Use "path" as its sole key.\n'
            "List every node as an integer, in order from start to goal. End your response with this JSON object."
        )
        return prompt.replace("\n\nNeighbors:", "\n" + output_format + "\n\nNeighbors:")
    prompt += (
        "\n\nThe controller maintains the confirmed path and provides environment updates.\n"
        "You may reason, then commit one next node using <action>integer_node_id</action>.\n"
        "Choose a legal unvisited next node from the latest environment update.\n"
        "Action tags and JSON path arrays are commitments, not tentative examples.\n"
        "After each committed node, the controller executes the move and adds an environment update to this context.\n"
        "Continue reasoning from the updated state and confirmed path.\n"
        "The trajectory is complete when the executed node reaches the goal."
    )
    if condition["prefer_minimum"]:
        prompt += "\nPrefer the legal next node with the smallest learned-map distance."
    return prompt


def detect_boundary(text, confirmed):
    """Detect the FIRST commitment, including a node inside an unfinished route.

    Integers need a delimiter: a streamed '2' must not be mistaken for '216'.
    Do not scan arbitrary numbers in reasoning or skip an invalid first action.
    """
    opening = OPENING.search(text)
    if opening is None:
        return None
    token = opening.group()
    if token == ENVIRONMENT_MARKER:
        return {"end": opening.end(), "error": "model_generated_environment", "style": "environment"}
    if token == "<action>":
        end = text.find("</action>", opening.end())
        if end < 0:
            return None
        value = text[opening.end():end].strip()
        event = {"end": end + len("</action>"), "style": "action"}
        if not re.fullmatch(r"0|[1-9][0-9]*", value):
            return {**event, "error": "invalid_action_format"}
        return {**event, "node": int(value)}
    cursor = opening.end()
    path = []
    while True:
        match = INTEGER.match(text, cursor)
        if match is None:
            # Only diagnose malformed JSON after a delimiter has arrived.
            delimiter = re.search(r"[,\]]", text[cursor:])
            if delimiter:
                return {"end": cursor + delimiter.end(), "style": "path", "error": "invalid_path_format"}
            return None
        path.append(int(match.group(1)))
        cursor = match.end()
        event = {"end": cursor, "style": "path"}
        if len(path) <= len(confirmed) and path[-1] != confirmed[len(path) - 1]:
            return {**event, "error": "changed_confirmed_path"}
        if len(path) == len(confirmed) + 1:
            return {**event, "node": path[-1]}
        if match.group(2) == "]":
            return {**event, "error": "no_new_node"}


def parse_final(text):
    answer = re.sub(r'(?:<\|im_end\|>|<\|endoftext\|>|<\|fim_suffix\|>)\s*$', '', text).strip()
    # A terminal Markdown fence is a presentation wrapper, not a different route.
    # Only accept a complete block containing one JSON object, with no trailing prose.
    match = FINAL_PATTERN.search(answer) or FINAL_FENCED_PATTERN.search(answer)
    if not match:
        raise ValueError("No final JSON object")
    value = json.loads(match.group(1))
    if not isinstance(value, dict) or set(value) != {"path"}:
        raise ValueError("Expected an object with only path")
    path = value["path"]
    if not isinstance(path, list) or not path or any(type(node) is not int for node in path):
        raise ValueError("path must be a nonempty integer array")
    return path


def path_error(case, path):
    if path[0] != case["start"]:
        return "wrong_start"
    if len(set(path)) != len(path):
        return "repeated_node"
    for left, right in zip(path, path[1:]):
        if right not in case["neighbors"].get(str(left), []):
            return "nonexistent_edge"
    if path[-1] != case["goal"]:
        return "goal_not_reached"
    return None


def map_update(case, confirmed, env, q_map):
    current = confirmed[-1]
    by_target = {int(env.actions[action, 1]): action for action in env.legal_actions(current)}
    choices = [(node, q_map.candidate_distance(current, by_target[node], case["goal"]))
               for node in case["neighbors"][str(current)] if node not in confirmed]
    lines = [
        "\n[Environment update]",
        f"Current node: {current}; goal: {case['goal']}; confirmed path: {json.dumps(confirmed)}",
        "Legal next nodes:",
        *[f"{node}, learned_map_distance_to_goal={distance:.6f}" for node, distance in choices],
        "Commit the next node using <action>integer_node_id</action>.",
        "[/Environment update]\n",
    ]
    return "\n".join(lines), choices


def run_trial(case, condition_name, condition, caller, env, q_map, budget, seed):
    prompt = trial_prompt(case, condition)
    transcript = caller.render(prompt)
    mapped = condition["map_updates"]
    confirmed = [case["start"]] if mapped else []
    record = {"protocol": PROTOCOL, "case_id": case["id"], "condition": condition_name,
              "start": case["start"], "goal": case["goal"], "shortest_moves": case["reference"]["length"],
              "prompt": prompt, "segments": [], "updates": [], "actions": [], "confirmed": confirmed,
              "final_path": None, "final_path_source": "environment" if mapped else "model",
              "failure": None, "generated_tokens": 0, "inserted_tokens": 0, "input_tokens": 0,
              "minimum_distance_choices": 0, "choice_count": 0, "requested_seed": seed,
              "seed_applied": caller.seed_applied, "token_accounting_complete": True}
    started = time.monotonic()
    while record["generated_tokens"] < budget:
        if mapped:
            if confirmed[-1] == case["goal"]:
                record["final_path"] = list(confirmed)
                break
            update, choices = map_update(case, confirmed, env, q_map)
            record["updates"].append(update)
            record["inserted_tokens"] += caller.count_tokens(update)
            transcript += update
            if not choices:
                record["failure"] = "dead_end"
                break
        remaining = budget - record["generated_tokens"]
        response = caller.generate(transcript, max_new_tokens=remaining,
                                   confirmed=confirmed if mapped else None,
                                   seed=seed + len(record["segments"]))
        record["segments"].append(response)
        record["generated_tokens"] += response["completion_tokens"]
        record["input_tokens"] += response["prompt_tokens"]
        if not response.get("token_accounting_complete", True):
            record["token_accounting_complete"] = False
            record["failure"] = "incomplete_token_accounting"
            break
        if not mapped:
            if response["finish_reason"] in ("length", "context"):
                record["failure"] = "context_limit" if response["finish_reason"] == "context" else "budget_truncated"
            else:
                try:
                    final = parse_final(response["text"])
                    record["final_path"] = final
                    record["failure"] = path_error(case, final)
                except ValueError as error:
                    record["failure"] = "missing_final_json"
                    record["parse_error"] = str(error)
            break
        boundary = detect_boundary(response["text"], confirmed)
        if boundary is None:
            record["failure"] = {"length": "budget_truncated", "context": "context_limit"}.get(
                response["finish_reason"], "missing_action")
            break
        response["boundary"] = boundary
        response["accepted_text"] = response["text"][:boundary["end"]]
        response["discarded_text"] = response["text"][boundary["end"]:]
        if "error" in boundary:
            record["failure"] = boundary["error"]
            break
        node = boundary["node"]
        if node in confirmed:
            record["failure"] = "repeated_node"
            break
        if node not in [target for target, _ in choices]:
            record["failure"] = "nonexistent_edge"
            break
        current = confirmed[-1]
        action_id = next(a for a in env.legal_actions(current) if int(env.actions[a, 1]) == node)
        actual = env.execute(current, action_id)
        record["actions"].append({"from": current, "to": actual, "action_id": int(action_id),
                                  "segment": len(record["segments"]) - 1, "style": boundary["style"]})
        record["choice_count"] += 1
        selected_distance = next(distance for target, distance in choices if target == node)
        if selected_distance <= min(distance for _, distance in choices) + 1e-9:
            record["minimum_distance_choices"] += 1
        confirmed.append(actual)
        transcript += response["accepted_text"]
        if actual == case["goal"]:
            record["final_path"] = list(confirmed)
            break
    if record["failure"] is None and record["final_path"] is None:
        record["failure"] = "budget_truncated"
    record["reached"] = record["failure"] is None
    record["shortest"] = record["reached"] and len(record["final_path"]) - 1 == record["shortest_moves"]
    record["elapsed_seconds"] = time.monotonic() - started
    return record


class TokenizerCaller:
    def render(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=self.enable_thinking)

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class TransformersContinuousCaller(TokenizerCaller):
    """Stop decoding at a node boundary, then re-encode the retained response."""

    seed_applied = True

    def __init__(self, model_path, *, enable_thinking, sampling, context_length, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

        self.torch = torch
        self.StoppingCriteria = StoppingCriteria
        self.StoppingCriteriaList = StoppingCriteriaList
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa").to(device).eval()
        self.enable_thinking = enable_thinking
        self.sampling = sampling
        self.context_length = context_length
        self.device = device

    def generate(self, transcript, *, max_new_tokens, confirmed, seed):
        torch = self.torch
        ids = self.tokenizer.encode(transcript, add_special_tokens=False, return_tensors="pt").to(self.device)
        prompt_tokens = int(ids.shape[1])
        room = self.context_length - prompt_tokens
        if room < 1:
            return {"text": "", "finish_reason": "context", "prompt_tokens": prompt_tokens,
                    "completion_tokens": 0}
        limit = min(max_new_tokens, room)
        stopping = None
        if confirmed is not None:
            tokenizer = self.tokenizer
            base = self.StoppingCriteria

            class StopAtNode(base):
                def __call__(self, input_ids, scores, **kwargs):
                    text = tokenizer.decode(input_ids[0, prompt_tokens:], skip_special_tokens=False)
                    return detect_boundary(text, confirmed) is not None

            stopping = self.StoppingCriteriaList([StopAtNode()])
        torch.manual_seed(seed)
        with torch.inference_mode():
            output = self.model.generate(
                ids, attention_mask=torch.ones_like(ids), max_new_tokens=limit, do_sample=True,
                **self.sampling, stopping_criteria=stopping, pad_token_id=self.tokenizer.eos_token_id)
        generated = output[0, prompt_tokens:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        if confirmed is not None and detect_boundary(raw, confirmed) is not None:
            finish = "controller"
        elif int(generated[-1]) in eos:
            finish = "stop"
        elif len(generated) >= limit:
            finish = "context" if room < max_new_tokens else "length"
        else:
            finish = "stop"
        return {"text": raw, "finish_reason": finish, "prompt_tokens": prompt_tokens,
                "completion_tokens": int(len(generated))}


class SGLangContinuousCaller(TokenizerCaller):
    """Parse cumulative SSE output and abort this request at the first node.

    The server may generate ahead before receiving abort. Drain its final usage,
    charge all reported tokens, and never put the excess text into context.
    """

    seed_applied = False  # SGLang 0.4.6 needs this fallback; 0.5.10 accepts sampling_seed.

    def __init__(self, model_path, endpoint, *, enable_thinking, sampling, context_length,
                 timeout_seconds=7200, per_request_seed=False):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.endpoint = endpoint.rstrip("/")
        self.enable_thinking = enable_thinking
        self.sampling = sampling
        self.context_length = context_length
        self.timeout_seconds = timeout_seconds
        self.seed_applied = per_request_seed

    def request(self, path, payload):
        request = urllib.request.Request(
            self.endpoint + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=self.timeout_seconds)

    def generate(self, transcript, *, max_new_tokens, confirmed, seed):
        ids = self.tokenizer.encode(transcript, add_special_tokens=False)
        room = self.context_length - len(ids)
        if room < 1:
            return {"text": "", "finish_reason": "context", "prompt_tokens": len(ids), "completion_tokens": 0}
        sampling = dict(self.sampling, max_new_tokens=min(max_new_tokens, room))
        if self.seed_applied:
            sampling["sampling_seed"] = seed
        mapped = confirmed is not None
        if mapped:
            sampling.update(stop=["</action>", ENVIRONMENT_MARKER], no_stop_trim=True, stream_interval=1)
        rid = uuid.uuid4().hex
        payload = {"input_ids": ids, "sampling_params": sampling, "rid": rid, "stream": mapped}
        boundary = None
        aborted = False
        final_seen = False
        raw = ""
        completion_tokens = 0
        meta = {}
        try:
            with self.request("/generate", payload) as response:
                if not mapped:
                    result = json.load(response)
                    raw, meta = result["text"], result["meta_info"]
                    completion_tokens = meta["completion_tokens"]
                    final_seen = meta.get("finish_reason") is not None
                else:
                    for line in response:
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            break
                        result = json.loads(data)
                        if "error" in result:
                            raise RuntimeError(result["error"])
                        text = result["text"]
                        if len(text) >= len(raw):
                            raw = text
                        meta = result["meta_info"]
                        completion_tokens = max(completion_tokens, meta["completion_tokens"])
                        final_seen = meta.get("finish_reason") is not None
                        if boundary is None:
                            boundary = detect_boundary(raw, confirmed)
                        if boundary is not None and not final_seen and not aborted:
                            with self.request("/abort_request", {"rid": rid}):
                                pass
                            aborted = True
        finally:
            if mapped and not final_seen and not aborted:
                with self.request("/abort_request", {"rid": rid}):
                    pass
        reason = (meta.get("finish_reason") or {}).get("type", "stream_ended")
        if boundary is not None:
            reason = "controller"
        elif reason == "length" and room < max_new_tokens:
            reason = "context"
        return {"text": raw, "finish_reason": reason, "backend_finish_reason": meta.get("finish_reason"),
                "prompt_tokens": meta.get("prompt_tokens", len(ids)), "completion_tokens": completion_tokens,
                "token_accounting_complete": final_seen, "request_id": rid, "abort_sent": aborted}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=["sglang", "transformers"], default="sglang")
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        parser.error("Config must select the controller_node_boundary protocol")
    condition = config["conditions"][args.condition]
    if args.model_path.name != f"Qwen3-4B-{condition['model'].capitalize()}-2507":
        parser.error("Condition and model path disagree")
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    cases = [suite["cases"][index] for index in config["case_indices"]]
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use a new output directory; existing pilot records must not be overwritten")
    env = environment_from_suite(suite)
    q_map = (GraphQMap.load(args.map, len(env.adjacency), len(env.actions)) if condition["map_updates"] else None)
    kwargs = dict(enable_thinking=condition["model"] == "thinking",
                  sampling=config["sampling"], context_length=config["context_length"])
    if args.backend == "sglang":
        caller = SGLangContinuousCaller(
            str(args.model_path), args.endpoint,
            per_request_seed=config.get("sglang_per_request_seed", False), **kwargs)
    else:
        caller = TransformersContinuousCaller(str(args.model_path), device=args.device, **kwargs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for index, case in zip(config["case_indices"], cases):
        result = run_trial(case, args.condition, condition, caller, env, q_map,
                           config["max_generated_tokens_per_question"], config["seed"] + index)
        result["run_inputs"] = {key: str(getattr(args, key)) for key in
                                ("config", "suite", "map", "model_path", "backend")}
        destination = args.out / f"{case['id']}.json"
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"saved": str(destination), "reached": result["reached"],
                          "failure": result["failure"], "segments": len(result["segments"]),
                          "generated_tokens": result["generated_tokens"]}), flush=True)


if __name__ == "__main__":
    main()
