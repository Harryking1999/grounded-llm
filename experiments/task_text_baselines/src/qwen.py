"""Evaluate the frozen prompts with a local Qwen model; task judges remain CPU-only."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

from .protocol import judge
from .run import write_json, summary


def final_text(tokens, tokenizer):
    marker = tokenizer.convert_tokens_to_ids("</think>")
    if marker in tokens:
        offset = len(tokens) - tokens[::-1].index(marker)
        return tokenizer.decode(tokens[offset:], skip_special_tokens=True).strip(), offset
    return tokenizer.decode(tokens, skip_special_tokens=True).strip(), None


def round_robin(cases):
    groups = defaultdict(list)
    for index, case in enumerate(cases):
        groups[case["task"]].append((index,case))
    return [group[offset] for offset in range(max(map(len,groups.values())))
            for group in groups.values() if offset < len(group)]


def main():
    parser = argparse.ArgumentParser(__doc__)
    for arg in ("suite", "config", "out", "model-path", "gpus"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if "vllm==" + version("vllm") != config["backend"]:
        raise ValueError("Backend version differs from formal contract")
    if len(gpus) != config["serving"]["tensor_parallel_size"]:
        raise ValueError("GPU count differs from formal TP contract")
    out = Path(args.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    if (out / "suite.json").resolve() != Path(args.suite).resolve():
        write_json(out / "suite.json",suite)
    records = [json.loads(p.read_text(encoding="utf-8")) for p in (out / "records").glob("*.json")]
    done = {r["case_id"] for r in records}
    state = {"status": "loading_model", "api": {"model": config["model"], "reasoning_effort": "Qwen thinking mode"},
             "config": config, "source_commit": args.source_commit,
             "runtime": {name:version(name) for name in ("vllm","torch","transformers")},
             "planned": len(suite["cases"]), "started_at": datetime.now(timezone.utc).isoformat(), "inflight": []}

    def save():
        state.update(summary=summary(records, suite["cases"]), updated_at=datetime.now(timezone.utc).isoformat())
        write_json(out / "progress.json",state)

    save()
    model = LLM(model=args.model_path, dtype=config["dtype"], trust_remote_code=False,
                enable_prefix_caching=True, disable_log_stats=True, **config["serving"])
    tokenizer = model.get_tokenizer()
    state["status"] = "running"
    queued = [(index,case) for index,case in round_robin(suite["cases"]) if case["id"] not in done]
    for batch_start in range(0,len(queued),config["batch_size"]):
        if (out / "stop_requested").exists():
            state["status"] = "stopped"
            break
        batch = queued[batch_start:batch_start+config["batch_size"]]
        texts, counts, params = [], [], []
        for index,case in batch:
            text = tokenizer.apply_chat_template([{"role":"user","content":case["prompt"]}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=config["enable_thinking"])
            count = len(tokenizer.encode(text,add_special_tokens=False))
            if count + config["max_output_tokens"] > config["serving"]["max_model_len"]:
                raise ValueError("Configured context cannot preserve the common output allowance")
            texts.append(text)
            counts.append(count)
            sampling = dict(config["sampling"])
            sampling["seed"] += index
            params.append(SamplingParams(max_tokens=config["max_output_tokens"], **sampling))
        state["inflight"] = [case["id"] for _,case in batch]
        save()
        started = time.monotonic()
        responses = model.generate(texts,params,use_tqdm=False)
        for (_,case), input_tokens, response in zip(batch,counts,responses):
            generated = response.outputs[0]
            tokens = list(generated.token_ids)
            raw, reasoning_tokens = final_text(tokens,tokenizer)
            verdict = judge(case,raw)
            if generated.finish_reason == "length":
                verdict = {"solved":False,"contract_pass":False,"failure":"truncated"}
            record = {"case_id":case["id"],"task":case["task"],"stratum":case["stratum"],"attempt":1,
                      "returned_model":config["model"],
                      "response_status":"completed" if generated.finish_reason != "length" else "incomplete",
                      "finish_reason":generated.finish_reason,"raw_output":raw,"full_output":tokenizer.decode(tokens,skip_special_tokens=False),
                      "usage":{"input_tokens":input_tokens,"output_tokens":len(tokens),"output_tokens_details":{"reasoning_tokens":reasoning_tokens}},
                      "verdict":verdict,"batch_elapsed_seconds":round(time.monotonic()-started,3)}
            write_json(out / "records" / (case["id"] + "_attempt1.json"), record)
            records.append(record)
            done.add(case["id"])
            print(json.dumps({"finished":len(records),"case":case["id"],"solved":verdict.get("solved"),"failure":verdict.get("failure"),"batch_seconds":record["batch_elapsed_seconds"]}),flush=True)
        state["inflight"] = []
        save()
    if len(done) == len(suite["cases"]):
        state["status"] = "completed"
    save()


if __name__ == "__main__":
    main()
