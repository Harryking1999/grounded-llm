"""Prepare replay-verified harder conditions from the existing task text."""
from collections import deque
import heapq
import json
import random

from experiments.qwen_path_blocks.src.prepare_path_suite import regular_connected_graph, shortest
from experiments.sol_dag_blocks.src.tasks import BlocksTask
from . import cube3, hanoi, sliding
from .prepare import make_hanoi
from grounded_llm.tasks.text import ROOT, prompt, section_text, judge

INVERSE = {"Up":"Down","Down":"Up","Left":"Right","Right":"Left"}


def graph_distances(graph,source):
    depths,queue = {source:0},deque([source])
    while queue:
        node = queue.popleft()
        for child in graph[node]:
            if child not in depths:
                depths[child] = depths[node] + 1
                queue.append(child)
    return depths


def uniform_tree(size,rng):
    sequence = [rng.randrange(size) for _ in range(size-2)]
    degree = [1]*size
    for node in sequence:
        degree[node] += 1
    leaves = [node for node in range(size) if degree[node] == 1]
    heapq.heapify(leaves)
    graph = {node:[] for node in range(size)}
    for node in sequence:
        leaf = heapq.heappop(leaves)
        graph[leaf].append(node)
        graph[node].append(leaf)
        degree[node] -= 1
        if degree[node] == 1:
            heapq.heappush(leaves,node)
    a,b = leaves
    graph[a].append(b)
    graph[b].append(a)
    return graph


def scramble(task,length,rng):
    env = cube3 if task == "cube3" else sliding
    goal = cube3.solved() if task == "cube3" else tuple(range(25 if task == 'puzzle24' else 16))
    state,actions,seen = goal,[],{goal}
    for _ in range(length):
        candidates = []
        for action in cube3.ACTIONS if task == "cube3" else INVERSE:
            try:
                child = env.step(state,action)
            except ValueError:
                continue
            if child not in seen:
                candidates.append((action,child))
        if not candidates:
            return scramble(task,length,rng)
        action,state = rng.choice(candidates)
        actions.append(action)
        seen.add(state)
    witness = [(a[:-1] if a.endswith("'") else a+"'") if task == "cube3" else INVERSE[a] for a in reversed(actions)]
    return {"task":task,"initial":list(state),"goal":list(goal),"stratum":f"scramble_{length}",
            "generation_scramble":actions,"reference":{"length":None,"upper_bound":length,"actions":witness},
            **({"manhattan_lower_bound":sliding.manhattan(state)} if task in ("puzzle15","puzzle24") else {})}


def blocks_case(options,count,rng):
    task = BlocksTask(options)
    for _ in range(1600):
        state,actions = 0,[]
        for _ in range(count):
            boundary = task.boundary(state)
            candidates = [(tile,action) for tile,action in task.placements
                          if not tile & state and (not state or tile & boundary)]
            if not candidates:
                break
            tile,action = rng.choice(candidates)
            state |= tile
            actions.append(action)
        if len(actions) == count:
            return {"task":"blocks","initial":state,"goal":0,"shape_options":options,
                    "stratum":f"SZ10_connected_{count}" if task.size == 10 else f"SZ10_board{task.size}_connected_{count}","generation_placements":count,
                    "reference":{"length":None,"upper_bound":count,"actions":actions}}
    raise ValueError("Could not construct requested connected blocks case")


def harder_prompt(case):
    task = case["task"]
    if task == "blocks":
        text = prompt(case)
        old = next(line for line in text.splitlines() if line.startswith("Shapes (/"))
        shapes = "; ".join(f"{i}={s}" for i,s in enumerate(case["shape_options"]["shape_rows"]))
        return text.replace(old,"Shapes (/ separates rows): "+shapes)
    if task in ("hanoi","path"):
        return prompt(case)
    if task in ("puzzle15","puzzle24"):
        size = 5 if task == 'puzzle24' else 4
        text = section_text("puzzle8","共享任务描述与规则")
        text = text.replace("3x3",f"{size}x{size}").replace("1 through 8",f"1 through {size*size-1}")
        text = text.replace("{initial_board_3_rows}",sliding.render(case["initial"]))
        text = text.replace("{goal_board_3_rows}",sliding.render(case["goal"]))
        return text+"\n\n"+section_text("puzzle8","SOTA 组／32B 组")
    text = section_text("cube2","共享任务描述与规则").split("Action examples:")[0]
    text = text.replace("2x2","3x3")
    text = text.replace("U, U', R, R', F, and F'.", "U, U', R, R', F, F', D, D', L, L', B, and B'.")
    text += "Action examples: each example starts independently from the state below.\nDo not apply these example turns in sequence.\n\n[Before each example]\n"+cube3.render(cube3.solved())
    for action in cube3.ACTIONS:
        text += "\n\n[After "+action+"]\n"+cube3.render(cube3.step(cube3.solved(),action))
    text += "\n\nInitial cube state:\n"+cube3.render(case["initial"])+"\n\nGoal cube state:\n"+cube3.render(case["goal"])
    contract = section_text("cube2","SOTA 组／32B 组")
    contract = contract.replace('"U", "U\'", "R", "R\'", "F", or "F\'"',
                                ', '.join(json.dumps(a) for a in cube3.ACTIONS[:-1])+', or '+json.dumps(cube3.ACTIONS[-1]))
    return text+"\n\n"+contract


def levels(task,options):
    return next(options[key] for key in ("construction_counts","scramble_lengths","disk_counts","node_counts") if key in options)


def prepare_level(task,level,options,config):
    rng = random.Random(config["seed"]+100000*list(config["tasks"]).index(task)+level*1000)
    count = config["cases_per_level"]
    cases,seen = [],set()
    value = levels(task,options)[level]
    if task == "path":
        if options.get('family') == 'uniform_tree':
            for _ in range(1000):
                graph = uniform_tree(value,rng)
                distances = graph_distances(graph,0)
                end = max(distances,key=distances.get)
                if max(graph_distances(graph,end).values()) >= options['minimum_distance']:
                    break
            else:
                raise ValueError('Unable to generate a tree with the configured distance floor')
        else:
            graph = regular_connected_graph(value,options["regular_degree"],rng)
        neighbors = {str(n):sorted(vs) for n,vs in graph.items()}
    if task == "hanoi":
        maximum = 2**value-1
        rows = make_hanoi({"disks":value,"goal_peg":"C","distance_ranges":[[maximum//2,maximum]],"per_range":count},rng)
    attempted = 0
    while len(cases) < count:
        attempted += 1
        if attempted > 100000:
            raise ValueError('Unable to sample the configured difficulty condition')
        if task == "blocks":
            options_source = json.loads((ROOT/options["shape_config"]).read_text())["blocks"]
            if 'grid_size' in options:
                options_source['grid_size'] = options['grid_size']
            case = blocks_case(options_source,value,rng)
        elif task in ("cube3","puzzle15","puzzle24"):
            case = scramble(task,value,rng)
            if task in ('puzzle15','puzzle24') and case['manhattan_lower_bound'] < options.get('minimum_manhattan',0):
                continue
        elif task == "hanoi":
            case = rows[len(cases)]
        else:
            source = rng.randrange(value)
            depths = graph_distances(graph,source)
            farthest = max(depths.values())
            if farthest < options.get('minimum_distance',0):
                continue
            goal = rng.choice([n for n,d in depths.items() if d == farthest])
            actions = shortest(graph,source,goal)
            family = '_tree' if options.get('family') == 'uniform_tree' else ''
            case = {"task":"path","initial":source,"goal":goal,"neighbors":neighbors,
                    "stratum":f"nodes_{value}{family}_eccentric_goal","reference":{"length":len(actions),"actions":actions}}
        identity = json.dumps([case["initial"],case["goal"]])
        if identity in seen:
            continue
        seen.add(identity)
        case["id"] = f"{task}_level{level}_{len(cases):02d}"
        case["level"] = level
        case["level_value"] = value
        case["prompt"] = harder_prompt(case)
        verdict = judge(case,json.dumps({"actions":case["reference"]["actions"],"final_status":"solved"}))
        if not verdict["contract_pass"]:
            raise ValueError((case["id"],verdict))
        cases.append(case)
    return cases
