"""Render authoritative task text and replay model plans, independently of API transport."""
import json
from pathlib import Path
import re
import importlib

import grounded_llm.tasks.blocks.rules as blocks
from . import cube3, puzzle15, sliding
from .cube2 import text as cube2
from .hanoi import text as hanoi
from .puzzle8 import rules as puzzle8

ROOT = Path(__file__).resolve().parents[2]


def section_text(task, heading):
    text = (ROOT / "tasks" / task / "prompts/README.md").read_text(encoding="utf-8")
    section = text.split("## " + heading, 1)[1].split("\n## ", 1)[0]
    return re.search(r"```text\n(.*?)\n```", section, re.S)[1]


def prompt(case):
    task = case["task"]
    text = section_text(task, "共享任务描述与规则")
    if task in ('bridge', 'sokoban', 'game24'):
        fields = importlib.import_module(f'grounded_llm.tasks.{task}.text').fields(case)
        fields['action_limit'] = case['action_limit']
    elif task == "blocks":
        if case.get('shape_options'):
            from grounded_llm.tasks.blocks.variants import BlocksTask
            rules = BlocksTask(case['shape_options'])
            text = text.replace('10x10',f'{rules.size}x{rules.size}')
            fields = {'initial_board_10_rows':'\n'.join(rules.to_rows(int(case['initial']))),
                      'goal_board_10_rows':'\n'.join(rules.to_rows(int(case['goal'])))}
        else:
            fields = {"initial_board_10_rows": blocks.to_grid(int(case["initial"])).replace("/", "\n"), "goal_board_10_rows": blocks.to_grid(int(case["goal"])).replace("/", "\n")}
    elif task == "path":
        fields = {"max_node_id": len(case["neighbors"]) - 1,
                  "adjacency_list": "\n".join(f"{i}: " + ", ".join(map(str, case["neighbors"][str(i)])) for i in range(len(case["neighbors"]))),
                  "initial_node": case["initial"], "goal_node": case["goal"]}
    elif task == "hanoi":
        fields = {"disk_count": len(case["initial"]), "initial_configuration": hanoi.render(case["initial"]), "goal_configuration": hanoi.render(case["goal"])}
    elif task == "cube2":
        fields = {"initial_cube_state": cube2.render(case["initial"]), "goal_cube_state": cube2.render(case["goal"])}
    else:
        fields = {"initial_board_3_rows": puzzle8.render(case["initial"]), "goal_board_3_rows": puzzle8.render(case["goal"])}
    for key, value in fields.items():
        text = text.replace("{" + key + "}", str(value))
    text += "\n\n" + section_text(task, "SOTA 组／32B 组")
    if re.search(r"\{[a-z_]+\}", text):
        raise ValueError("Unrendered prompt placeholder")
    return text


def parse(raw):
    try:
        output = json.loads(raw)
        return output, True
    except (ValueError, TypeError):
        try:
            return json.loads(raw[raw.index("{"):raw.rindex("}") + 1]), False
        except (ValueError, TypeError, AttributeError):
            return None, False


def judge(case, raw):
    output, strict = parse(raw)
    if not isinstance(output, dict) or not isinstance(output.get("actions"), list):
        return {"solved": False, "contract_pass": False, "failure": "parse_error", "strict_json": strict}
    task, actions = case["task"], output["actions"]
    if task in ('bridge', 'sokoban', 'game24'):
        return judge_plan(case, output, strict)
    state = tuple(case["initial"]) if task in ("hanoi", "puzzle8", "puzzle15", "puzzle24", "cube3") else case["initial"]
    goal = tuple(case["goal"]) if task in ("hanoi", "puzzle8", "puzzle15", "puzzle24", "cube3") else case["goal"]
    shape_rules = None
    if task == "blocks" and case.get("shape_options"):
        from grounded_llm.tasks.blocks.variants import BlocksTask
        shape_rules = BlocksTask(case["shape_options"])
    illegal, detail, executed = None, None, 0
    reports_valid, reports_correct = 0, 0
    for index, action in enumerate(actions):
        try:
            if task == "blocks":
                state = shape_rules.apply(int(state),action) if shape_rules else blocks.apply(int(state), action)
            elif task == "path":
                if not isinstance(action, dict) or set(action) != {"from", "to"} or any(type(action[k]) is not int for k in ("from", "to")):
                    raise ValueError("Expected integer from/to object")
                if action["from"] != state or action["to"] not in case["neighbors"][str(state)]:
                    raise ValueError("Not a current-node edge")
                state = action["to"]
            else:
                state = {"hanoi": hanoi.step, "cube2": cube2.step, "cube3":cube3.step,
                         "puzzle8": puzzle8.step, "puzzle15":puzzle15.step, "puzzle24":sliding.step}[task](state, action)
            executed += 1
        except (ValueError, TypeError, KeyError, IndexError) as error:
            illegal, detail = index, str(error)
            break
        if task == 'blocks' and case.get('state_report_required'):
            expected = shape_rules.to_rows(int(state)) if shape_rules else blocks.to_grid(int(state)).split('/')
            reported = action.get('board_after')
            valid = (isinstance(reported,list) and len(reported) == len(expected) and
                     all(isinstance(row,str) and len(row) == len(expected[0]) and not set(row)-{'0','1'} for row in reported))
            reports_valid += valid
            reports_correct += valid and reported == expected
    solved = illegal is None and state == goal
    status = output.get("final_status")
    status_correct = status == ("solved" if solved else "unsolved")
    minimum = case["reference"]["length"]
    result = {"solved": solved, "contract_pass": solved and strict and status_correct,
            "strict_json": strict, "status_correct": status_correct, "reported_status": status,
            "failure": "illegal_action" if illegal is not None else None if solved else "unsolved",
            "first_illegal_index": illegal, "illegal_detail": detail, "executed": executed,
            "plan_length": len(actions), "optimal": solved and len(actions) == minimum if minimum is not None else None,
            "excess_moves": len(actions) - minimum if solved and minimum is not None else None}
    if task == 'blocks' and case.get('state_report_required'):
        result.update(state_reports_evaluated=executed,state_reports_valid=reports_valid,
                      state_reports_correct=reports_correct)
        result['contract_pass'] = result['contract_pass'] and reports_correct == executed
    return result


def judge_plan(case, output, strict):
    """Use task-owned transitions with the same completion/format split as older tasks."""
    adapter = importlib.import_module(f'grounded_llm.tasks.{case["task"]}.text')
    state, total, executed, illegal, detail = adapter.initial(case), 0, 0, None, None
    actions = output['actions']
    for index, action in enumerate(actions):
        try:
            state, cost = adapter.step(case, state, action)
            total += cost
            executed += 1
        except (ValueError, TypeError, KeyError, IndexError) as error:
            illegal, detail = index, str(error)
            break
    length_valid = len(actions) <= case['action_limit']
    if case['task'] == 'game24':
        length_valid = len(actions) == 3
    solved = illegal is None and length_valid and adapter.solved(case, state)
    status = output.get('final_status')
    status_correct = status == ('solved' if solved else 'unsolved')
    optimum = case['reference']['cost']
    return {'solved': solved, 'contract_pass': solved and strict and status_correct,
            'strict_json': strict, 'status_correct': status_correct, 'reported_status': status,
            'failure': 'illegal_action' if illegal is not None else 'action_limit' if not length_valid else None if solved else 'unsolved',
            'first_illegal_index': illegal, 'illegal_detail': detail, 'executed': executed,
            'plan_length': len(actions), 'cost': total, 'optimal': solved and total == optimum,
            'excess_cost': total - optimum if solved else None}
