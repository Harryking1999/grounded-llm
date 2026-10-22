"""Produce a compact, current results report from saved records, without API calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from .run import summary, write_json
from .budget import output_limit, rescore, POLICY

NAMES = {"blocks":"积木", "path":"寻路", "hanoi":"汉诺塔", "cube2":"二阶魔方", "puzzle8":"8-Puzzle",
         "cube3":"三阶魔方", "puzzle15":"4×4 华容道", "puzzle24":"5×5 华容道",
         "bridge":"手电过桥", "sokoban":"推箱子", "game24":"24点",
         "rush_hour":"Rush Hour", "token_swapping":"Token Swapping"}


def build(run, limit=None):
    run = Path(run)
    progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
    suite = json.loads((run / "suite.json").read_text(encoding="utf-8"))
    attempts = [json.loads(p.read_text(encoding="utf-8")) for p in (run / "records").glob("*.json")]
    latest = {}
    for row in sorted(attempts, key=lambda r:(r["case_id"],r["attempt"])):
        latest[row["case_id"]] = row
    records = list(latest.values())
    execution_records = [dict(r,verdict=r.get('execution_verdict',r['verdict'])) for r in records]
    if limit is not None:
        records = [rescore(r,limit) for r in records]
    valid = [row for row in records if not row.get('service_error')]
    metadata = {'valid_answers':len(valid),
                'model_reported':sum(row.get('returned_model') is not None for row in valid),
                'effort_counts':dict(Counter((row.get('returned_reasoning') or {}).get('effort') or 'not_returned' for row in valid)),
                'effort_reported_on_reused_answers':sum(bool(row.get('reused_from')) and bool((row.get('returned_reasoning') or {}).get('effort')) for row in valid)}
    results = summary(records,suite["cases"])
    execution = summary(execution_records,suite['cases'])
    for task,row in results.items():
        row['execution_solved'] = execution[task]['solved']
        row['budget_exceeded'] = sum(r.get('budget_status') == 'exceeded' for r in records if r['task'] == task)
        row['budget_unknown'] = sum(r.get('budget_status') == 'unknown' for r in records if r['task'] == task)
    result = {"status":progress["status"], "model_request":progress["api"], "tasks":results,
              "recorded_attempts":len(attempts), "service_error_attempts":sum(bool(r.get("service_error")) for r in attempts),
              "returned_model_counts":dict(Counter(str(r.get("returned_model")) for r in records)),
              "exceeded_output_limit":sum(r.get("exceeds_requested_budget") is True for r in records),
              "run_path":str(run), "updated_at":progress["updated_at"],"response_metadata_coverage":metadata}
    if limit is not None:
        result['budget_policy'] = {'max_output_tokens':limit,'scope':'reasoning_plus_final_answer',
            'exceeded':sum(r.get('budget_status') == 'exceeded' for r in valid),
            'unknown':sum(r.get('budget_status') == 'unknown' for r in valid),
            'request_limit_counts':dict(Counter(str(r.get('requested_max_output_tokens',suite['config'].get('api',{}).get('max_output_tokens',progress['api'].get('max_output_tokens')))) for r in valid))}
    adaptive = suite["config"].get("study") == "adaptive_complete_plan_difficulty"
    new_tasks = suite["config"].get("study") == "new_tasks_flash_fixed_setting_screening"
    rush_token = suite['config'].get('study') == 'rush_token_flash_fixed_screening'
    exact_goals = suite['config'].get('study') == 'specified_complete_goal_flash_screening'
    if rush_token or exact_goals:
        by_id = {c['id']:c for c in suite['cases']}
        for task, values in results.items():
            solved_rows = [r for r in valid if r['task'] == task and r['verdict'].get('solved')]
            costs = [r['verdict']['cost'] for r in solved_rows]
            references = [by_id[r['case_id']]['reference']['cost'] for r in solved_rows]
            gaps = [c-d for c,d in zip(costs,references)]
            ratios = [c/d for c,d in zip(costs,references)]
            values['cost_quality'] = dict(scope='successful answers only', count=len(costs),
                mean_cost=statistics.mean(costs) if costs else None,
                mean_reference_cost=statistics.mean(references) if costs else None,
                mean_excess=statistics.mean(gaps) if gaps else None,
                max_excess=max(gaps) if gaps else None,
                mean_cost_ratio=statistics.mean(ratios) if ratios else None)
    first_error = suite['config'].get('policy') == 'first_valid_model_error_per_task'
    policy = suite.get('current_policy',suite['config'])
    screening,confirmation = policy.get('cases_per_level',5),policy.get('confirmation_cases',10)
    lines = ["# 指定完整目标纯文字筛查" if exact_goals else "# Rush Hour与Token Swapping纯文字筛查" if rush_token else "# 三个新任务首次错误探索" if first_error else "# 逐项加难试验" if adaptive else "# 三个新任务纯文字筛查" if new_tasks else "# 五任务纯文字完整规划基线", "",
             f"状态：{progress['status']}。模型请求：`{progress['api']['model']}`；推理设置：`{progress['api']['reasoning_effort']}`。",
             "", f"每档至多{screening}题；每任务出现首个有效未完成即停止，不以单个失败估计成功率。服务错误不算模型失败。" if first_error else f"每档先测 {screening} 题，成功率不高于一半时用同档 {confirmation} 道独立新题确认；仍不高于一半才标记找到困难条件。格式问题单列，打乱步数不等于最短距离。" if adaptive else f"冻结题集共 {len(suite['cases'])} 题，一次输出完整计划，无工具、无中途反馈。主指标为环境回放到达目标，最优解和输出格式另计。",
             "", "| 任务 | 有效答卷／计划 | 待补服务失败 | 解出 | 严格合同通过 | 最优解 |", "|---|---:|---:|---:|---:|---:|"]
    for task, values in results.items():
        optimal = str(values['optimal']) if values['optimal_evaluated'] else '—'
        lines.append(f"| {NAMES[task]} | {values['valid_responses']}/{values['planned']} | {values['service_errors']} | {values['solved']} | {values['contract_pass']} | {optimal} |")
    if adaptive:
        lines += ["", "## 当前难度与独立确认", ""]
        for task,row in progress.get("tasks",{}).items():
            if first_error and row['status'] == 'model_error_found':
                found = row['first_error']
                lines.append(f"- {NAMES[task]}：在{found['level_value']}档发现首个失败 `{found['case_id']}`；{found['verdict'].get('failure')}。")
                continue
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
    if limit is not None:
        lines += [f'输出评分上限 {limit:,} token，包含思考与最终答案；超限即错，缺失用量不计预算内成功。历史答卷按新规则重算，不代表重新在该上限下生成。','']
    for task, values in results.items():
        failures = {k:v for k,v in values["failures"].items() if k != "success"}
        lines.append(f"- {NAMES[task]}：" + ("；".join(f"{k}={v}" for k,v in failures.items()) or "已返回题暂无失败。"))
    lines += [f"- 服务错误尝试共 {result['service_error_attempts']} 次；只重试基础设施失败，模型答错和截断不重采样。",
              f"- 最新答卷中 {result['exceeded_output_limit']} 份超出请求的输出 token 上限；保留执行结果，但不能视为严格预算内成功。",
              f"- 已记录输出 token 总数：{sum((r.get('usage') or {}).get('output_tokens') or 0 for r in attempts):,}。",
              "- 第三方接口未提供返回模型名称与 reasoning 配置时，只能确认提交了请求参数，不能独立核验后台身份或强度。",
              "- 仅评指定完整目标。推箱子和Rush Hour沿用已测原始起点并明确见证终局，Token Swapping在固定树上同时重标起终点棋子。已核对精确成本；这批定向诊断不估计独立总体成功率，也不把旧完成规则成绩混入。" if exact_goals else "- Rush Hour为8个公开无墙分量各5个起点，包含最短轨迹中间状态；Token Swapping为tket公开完整映射实例，精确最优值由本地搜索重算，上游交换序列仅为上界。不是代表性总体成功率。" if rush_token else "- 本轮加难是寻找失败点的适应性小样本试验，不估计整体成功率；完整计划，无工具、无中途反馈。" if adaptive else "- 复用既定Qmap留出条件；推箱子按布局等额并覆盖距离范围抽取，24点来自XoT测试源。结果只描述本批筛查题，不代表标准Boxoban或全部任务分布。" if new_tasks else "- 积木按 12 次不重叠放置构造，最短移除数可能更少；其余按精确距离分层。结果只描述本轮题集。",
              "- 配置与原始逐题结果保留在运行目录；研究参数见本实验 configs/，不从本表反推全部运行设置。", ""]
    return result,"\n".join(lines)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument('--budget-config',type=Path,default=POLICY)
    args=parser.parse_args()
    result,text=build(args.run,output_limit(args.budget_config))
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out / "summary.json",result)
    (out / "report.md").write_text(text,encoding="utf-8",newline="\n")
    print(json.dumps({"status":result["status"],"tasks":{k:{"finished":v['finished'],"solved":v['solved']} for k,v in result['tasks'].items()}}))


if __name__ == "__main__":
    main()
