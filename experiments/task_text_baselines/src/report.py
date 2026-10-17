"""Produce a compact, current results report from saved records, without API calls."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .run import summary, write_json

NAMES = {"blocks":"积木", "path":"寻路", "hanoi":"汉诺塔", "cube2":"二阶魔方", "puzzle8":"8-Puzzle",
         "cube3":"三阶魔方", "puzzle15":"4×4 华容道"}


def build(run):
    run = Path(run)
    progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
    suite = json.loads((run / "suite.json").read_text(encoding="utf-8"))
    attempts = [json.loads(p.read_text(encoding="utf-8")) for p in (run / "records").glob("*.json")]
    latest = {}
    for row in sorted(attempts, key=lambda r:(r["case_id"],r["attempt"])):
        latest[row["case_id"]] = row
    records = list(latest.values())
    results = summary(records,suite["cases"])
    result = {"status":progress["status"], "model_request":progress["api"], "tasks":results,
              "recorded_attempts":len(attempts), "service_error_attempts":sum(bool(r.get("service_error")) for r in attempts),
              "returned_model_counts":dict(Counter(str(r.get("returned_model")) for r in records)),
              "exceeded_output_limit":sum(r.get("exceeds_requested_budget") is True for r in records),
              "run_path":str(run), "updated_at":progress["updated_at"]}
    adaptive = suite["config"].get("study") == "adaptive_complete_plan_difficulty"
    lines = ["# 逐项加难试验" if adaptive else "# 五任务纯文字完整规划基线", "",
             f"状态：{progress['status']}。模型请求：`{progress['api']['model']}`；推理设置：`{progress['api']['reasoning_effort']}`。",
             "", "每档先测 5 题，成功率不高于一半时用同档 10 道独立新题确认；仍不高于一半才标记找到困难条件。格式问题单列，打乱步数不等于最短距离。" if adaptive else "同一套各 40 题，一次输出完整计划，无工具、无中途反馈。主指标为环境回放到达目标，最优解和输出格式另计。",
             "", "| 任务 | 有效答卷／计划 | 待补服务失败 | 解出 | 严格合同通过 | 最优解 |", "|---|---:|---:|---:|---:|---:|"]
    for task, values in results.items():
        optimal = str(values['optimal']) if values['optimal_evaluated'] else '—'
        lines.append(f"| {NAMES[task]} | {values['valid_responses']}/{values['planned']} | {values['service_errors']} | {values['solved']} | {values['contract_pass']} | {optimal} |")
    if adaptive:
        lines += ["", "## 当前难度与独立确认", ""]
        for task,row in progress.get("tasks",{}).items():
            if 'level_results' in row:
                label = {'active':'继续加难／确认','difficulty_confirmed':'已找到困难条件','range_exhausted':'当前档位耗尽，尚未确认困难条件','service_blocked':'服务失败待补'}.get(row['status'],row['status'])
                done = '；'.join(f"{r['level_value']} 档 {'确认' if r['phase']=='confirm' else '初测'} {r['solved']}/{r['valid']}" for r in row['level_results'])
                lines.append(f"- {NAMES[task]}：{label}。{done}")
                continue
            failure = row.get("first_error")
            if not failure:
                label = {"active":"继续测试", "range_exhausted":"当前范围全部通过", "service_blocked":"连接失败，待补测"}.get(row['status'],row['status'])
                lines.append(f"- {NAMES[task]}：{label}。")
                continue
            verdict = failure['verdict']
            category = {"illegal_action":"非法动作", "unsolved":"计划未到目标", "parse_error":"答案无法解析", "truncated":"输出达到预算上限"}.get(verdict.get('failure'),"输出格式未通过")
            index = verdict.get('first_illegal_index')
            detail = f"第 {index+1} 步{category}" if index is not None else category
            origin = "原 40 题" if failure['origin'] == 'original_baseline' else '加难试验'
            lines.append(f"- {NAMES[task]}：{origin} `{failure['case_id']}` 出现{detail}，已停止升档。")
    lines += ["", "## 难度分层", "", "| 任务 | 分层 | 有效答卷 | 待补服务失败 | 解出 |", "|---|---|---:|---:|---:|"]
    for task, values in results.items():
        for name, row in sorted(values["strata"].items()):
            lines.append(f"| {NAMES[task]} | {name} | {row['valid_responses']} | {row['service_errors']} | {row['solved']} |")
    lines += ["", "## 失败与解释边界", ""]
    for task, values in results.items():
        failures = {k:v for k,v in values["failures"].items() if k != "success"}
        lines.append(f"- {NAMES[task]}：" + ("；".join(f"{k}={v}" for k,v in failures.items()) or "已返回题暂无失败。"))
    lines += [f"- 服务错误尝试共 {result['service_error_attempts']} 次；只重试基础设施失败，模型答错和截断不重采样。",
              f"- 最新答卷中 {result['exceeded_output_limit']} 份超出请求的输出 token 上限；保留执行结果，但不能视为严格预算内成功。",
              f"- 已记录输出 token 总数：{sum((r.get('usage') or {}).get('output_tokens') or 0 for r in attempts):,}。",
              "- 第三方接口未提供返回模型名称与 reasoning 配置时，只能确认提交了请求参数，不能独立核验后台身份或强度。",
              "- 本轮加难是寻找失败点的适应性小样本试验，不估计整体成功率；完整计划，无工具、无中途反馈。" if adaptive else "- 积木按 12 次不重叠放置构造，最短移除数可能更少；其余按精确距离分层。结果只描述本轮题集。",
              "- 配置与原始逐题结果保留在运行目录；研究参数见本实验 configs/，不从本表反推全部运行设置。", ""]
    return result,"\n".join(lines)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    result,text=build(args.run)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out / "summary.json",result)
    (out / "report.md").write_text(text,encoding="utf-8",newline="\n")
    print(json.dumps({"status":result["status"],"tasks":{k:{"finished":v['finished'],"solved":v['solved']} for k,v in result['tasks'].items()}}))


if __name__ == "__main__":
    main()
