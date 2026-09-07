"""Replay every final sample and summarize outcomes and empirical shape distributions."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tasks import ROOT, STUDY, TASKS
from run import evaluate, transport


def distribution(counts, labels):
    total = sum(counts.values())
    return {"total": total, "counts": {str(k): counts[k] for k in labels},
            "probabilities": {str(k): counts[k] / total if total else None for k in labels}}


def summarize(run, suite):
    cases = {c["id"]: c for c in suite["cases"]}
    keys = [(r["case_id"], r["replicate"]) for r in run["cases"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate trial slots")
    for record in run["cases"]:
        if record.get("response_status") == "completed" and not record.get("api_error"):
            replay = evaluate(cases[record["case_id"]], record["raw_output"])
            if replay != record["verdict"]:
                raise ValueError(f"Replay mismatch: {record['case_id']} replicate {record['replicate']}")
    result = {"source_commit": run["source_commit"], "run_status": run["status"], "api_config": run["api_config"],
              "service_failures_archived": len(run.get("prior_failed_attempts", [])), "conditions": {}}
    for condition, task in TASKS.items():
        selected = [c for c in cases.values() if c["condition"] == condition]
        rows = [r for r in run["cases"] if r["condition"] == condition]
        final = [r for r in rows if transport.final_sample(r)]
        complete = [r for r in final if r.get("response_status") == "completed"]
        per_case = {}
        for case in selected:
            group = [r for r in final if r["case_id"] == case["id"]]
            per_case[case["id"]] = {"n": len(group), "successes": sum(r["verdict"].get("pass", False) for r in group),
                                   "contract_successes": sum(r["verdict"].get("contract_pass", False) for r in group),
                                   "complete": len(group) == case["replicates"]}
        eligible = [v for v in per_case.values() if v["complete"]]
        success = sum(r["verdict"].get("pass", False) for r in final)
        summary = {"planned_samples": sum(c["replicates"] for c in selected), "final_samples": len(final),
                   "completed_responses": len(complete), "budget_truncations": sum(transport.budget_truncated(r) for r in final),
                   "current_service_failures": len(rows) - len(final), "successes": success,
                   "sample_success_rate": success / len(final) if final else None,
                   "contract_successes": sum(r["verdict"].get("contract_pass", False) for r in final),
                   "execution_successes": sum(r["verdict"].get("execution_pass", False) for r in final),
                   "pass8_eligible_cases": len(eligible), "pass8_solved_cases": sum(c["successes"] > 0 for c in eligible),
                   "pass_at_8": sum(c["successes"] > 0 for c in eligible) / len(eligible) if eligible else None,
                   "zero_of_8_cases": [k for k, v in per_case.items() if v["complete"] and v["successes"] == 0],
                   "failure_counts": dict(Counter(r["verdict"].get("failure_type") or "success" for r in final)),
                   "illegal_answers": sum(bool(r["verdict"].get("illegal_action") or r["verdict"].get("illegal_move")) for r in complete),
                   "illegal_answer_rate_completed": sum(bool(r["verdict"].get("illegal_action") or r["verdict"].get("illegal_move")) for r in complete) / len(complete) if complete else None,
                   "first_illegal_reasons": dict(Counter((r["verdict"].get("illegal_action") or r["verdict"].get("illegal_move"))["reason"] for r in complete if r["verdict"].get("illegal_action") or r["verdict"].get("illegal_move"))),
                   "input_tokens": sum(r.get("input_tokens") or 0 for r in final),
                   "output_tokens": sum(r.get("output_tokens") or 0 for r in final),
                   "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in final),
                   "returned_models": dict(Counter(r.get("response", {}).get("model", "missing") for r in final)),
                   "returned_reasoning_settings": dict(Counter(json.dumps(r.get("response", {}).get("reasoning"), sort_keys=True) for r in final)),
                   "returned_output_limits": dict(Counter(str(r.get("response", {}).get("max_output_tokens")) for r in final)),
                   "per_case": per_case}
        if condition.startswith("blocks"):
            traces = [t for r in complete for t in r["verdict"].get("trace", [])]
            n = len(traces)
            summary["state_reports"] = {"evaluated_legal_prefix_steps": n,
                "correct": sum(t["state_report_correct"] for t in traces),
                "valid": sum(t["state_report_valid"] for t in traces),
                "exact_accuracy": sum(t["state_report_correct"] for t in traces) / n if n else None,
                "cell_accuracy_missing_as_wrong": sum(t["cell_matches"] for t in traces) / (100 * n) if n else None}
            counts = {k: Counter() for k in ("input_construction", "output_all_parsed", "output_legal_prefix", "output_successful")}
            for case in selected:
                counts["input_construction"].update(a["shape_id"] for a in case["construction_reference"])
            conditional = {sid: {"eligible_trials": 0, "output_contains_shape": 0} for sid in range(len(task.shapes))}
            unknown = 0
            for record in complete:
                output, _ = transport.parse_output(record["raw_output"])
                actions = output.get("actions", []) if isinstance(output, dict) else []
                actions = actions if isinstance(actions, list) else []
                ids = [a.get("shape_id") if isinstance(a, dict) else None for a in actions]
                valid_ids = [sid for sid in ids if type(sid) is int and 0 <= sid < len(task.shapes)]
                unknown += len(ids) - len(valid_ids)
                counts["output_all_parsed"].update(valid_ids)
                counts["output_legal_prefix"].update(t["action"]["shape_id"] for t in record["verdict"].get("trace", []))
                if record["verdict"].get("pass"):
                    counts["output_successful"].update(valid_ids)
                for sid in {a["shape_id"] for a in cases[record["case_id"]]["construction_reference"]}:
                    conditional[sid]["eligible_trials"] += 1
                    conditional[sid]["output_contains_shape"] += sid in valid_ids
            for entry in conditional.values():
                entry["probability"] = entry["output_contains_shape"] / entry["eligible_trials"] if entry["eligible_trials"] else None
            summary["shapes"] = {"by_id": {}, "by_subclass": {}, "unknown_output_shape_ids": unknown,
                                 "output_presence_given_input_presence_completed_responses": conditional}
            for name, values in counts.items():
                summary["shapes"]["by_id"][name] = distribution(values, range(len(task.shapes)))
                subclasses = Counter()
                for sid, count in values.items():
                    subclasses[task.subclasses[sid]] += count
                summary["shapes"]["by_subclass"][name] = distribution(subclasses, sorted(set(task.subclasses)))
        else:
            summary["shortest_length_counts"] = dict(Counter(c["reference"]["length"] for c in selected))
        result["conditions"][condition] = summary
    return result


def markdown(summary, run_path):
    lines = ["# Sol medium 实验结果", "", f"原始运行：`{run_path}`；运行源码 commit：`{summary['source_commit']}`。", "",
             "| 条件 | 已完成试验 | 任务成功 | 完整合同通过 | pass@8 | 截断 | 非法答案 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, s in summary["conditions"].items():
        lines.append(f"| {name} | {s['final_samples']}/{s['planned_samples']} | {s['successes']} | {s['contract_successes']} | {s['pass8_solved_cases']}/{s['pass8_eligible_cases']} | {s['budget_truncations']} | {s['illegal_answers']} |")
    lines += ["", "pass@8 仅在八次均已有最终试验记录的实例上计算；预算截断计失败。完整合同通过额外要求严格 JSON、正确最终状态与积木每一步完整棋盘报告。非法答案计数限完整响应。", ""]
    for name, s in summary["conditions"].items():
        lines += [f"## {name}", "", f"失败分类：`{json.dumps(s['failure_counts'], ensure_ascii=False)}`。八次全失败实例：`{s['zero_of_8_cases']}`。", ""]
        if "state_reports" in s:
            state = s["state_reports"]
            lines += [f"合法前缀逐步棋盘完全正确：{state['correct']}/{state['evaluated_legal_prefix_steps']}；格式有效：{state['valid']}/{state['evaluated_legal_prefix_steps']}。缺失报告计错，非法动作及后续不评价。", "",
                      "| shape_id | mask | 子类 | 输入构造计数 | 输出全部可解析动作 | 合法前缀动作 | 成功答案动作 |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
            task = TASKS[name]
            for sid, shape in enumerate(task.shapes):
                cols = [s["shapes"]["by_id"][k]["counts"][str(sid)] for k in ("input_construction", "output_all_parsed", "output_legal_prefix", "output_successful")]
                lines.append(f"| {sid} | `{shape}` | {task.subclasses[sid]} | " + " | ".join(map(str, cols)) + " |")
            lines += ["", "归一化频率、按子类分布和输入含该形状时的输出出现率见 `summary.json`。输入分解不唯一，这些是经验选择频率，不是识别准确率。", ""]
    lines += ["## 限制", "", "本轮只有 Sol 的一次性完整计划基线；形状集合、提示词、样本和网关均与旧轮不同，不能用差值证明状态增强或模型优劣。路径按至少四步筛选，不能代表任意随机点对。逐步棋盘自报来自模型，没有环境反馈；报告正确不等于搜索或规划机制正确。第三方网关返回名称只作为元数据，不证明底层模型身份。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default=str(STUDY / "results"))
    args = parser.parse_args()
    run = json.loads((ROOT / args.run).read_text(encoding="utf-8"))
    suite = json.loads((ROOT / run["suite_path"]).read_text(encoding="utf-8"))
    summary = summarize(run, suite)
    summary["run_path"] = args.run
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(markdown(summary, args.run), encoding="utf-8")
    print(json.dumps({k: {f: v[f] for f in ("final_samples", "successes", "contract_successes", "pass_at_8")} for k, v in summary["conditions"].items()}))
