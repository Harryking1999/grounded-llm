"""Render authoritative task text and replay model plans, independently of API transport."""
import json
from pathlib import Path
import re

from experiments.gcml_counterexamples.src import blocks
from . import cube2, hanoi, puzzle8

ROOT = Path(__file__).resolve().parents[3]


def section_text(task, heading):
    text = (ROOT / "tasks" / task / "prompts/README.md").read_text(encoding="utf-8")
    section = text.split("## " + heading, 1)[1].split("\n## ", 1)[0]
    return re.search(r"```text\n(.*?)\n```", section, re.S)[1]


def prompt(case):
    task = case["task"]
    text = section_text(task, "共享任务描述与规则")
    if task == "blocks":
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
    state = tuple(case["initial"]) if task in ("hanoi", "puzzle8") else case["initial"]
    goal = tuple(case["goal"]) if task in ("hanoi", "puzzle8") else case["goal"]
    illegal, detail, executed = None, None, 0
    for index, action in enumerate(actions):
        try:
            if task == "blocks":
                state = blocks.apply(int(state), action)
            elif task == "path":
                if not isinstance(action, dict) or set(action) != {"from", "to"} or any(type(action[k]) is not int for k in ("from", "to")):
                    raise ValueError("Expected integer from/to object")
                if action["from"] != state or action["to"] not in case["neighbors"][str(state)]:
                    raise ValueError("Not a current-node edge")
                state = action["to"]
            else:
                state = {"hanoi": hanoi.step, "cube2": cube2.step, "puzzle8": puzzle8.step}[task](state, action)
            executed += 1
        except (ValueError, TypeError, KeyError, IndexError) as error:
            illegal, detail = index, str(error)
            break
    solved = illegal is None and state == goal
    status = output.get("final_status")
    status_correct = status == ("solved" if solved else "unsolved")
    minimum = case["reference"]["length"]
    return {"solved": solved, "contract_pass": solved and strict and status_correct,
            "strict_json": strict, "status_correct": status_correct, "reported_status": status,
            "failure": "illegal_action" if illegal is not None else None if solved else "unsolved",
            "first_illegal_index": illegal, "illegal_detail": detail, "executed": executed,
            "plan_length": len(actions), "optimal": solved and len(actions) == minimum,
            "excess_moves": len(actions) - minimum if solved else None}
