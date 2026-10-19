"""Study-specific environments and judges. No evaluator tools are sent to the model."""
from grounded_llm.tasks.blocks.variants import BlocksTask
from collections import deque
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
CONFIG = json.loads((STUDY / "configs/sol_medium.json").read_text())




class Blocks8Task(BlocksTask):
    construction_objects = 8
    condition = "blocks8"


class Blocks12Task(BlocksTask):
    construction_objects = 12
    condition = "blocks12"


class DagTask:
    condition = "path_dag"
    edge_error = "nonexistent_directed_edge"

    def shortest(self, case):
        queue, parents = deque([case["start"]]), {case["start"]: None}
        while queue:
            node = queue.popleft()
            if node == case["goal"]:
                route = []
                while parents[node] is not None:
                    before = parents[node]
                    route.append({"from": before, "to": node})
                    node = before
                return list(reversed(route))
            for other in case["neighbors"][str(node)]:
                if other not in parents:
                    parents[other] = node
                    queue.append(other)
        return None

    def prompt(self, case):
        adjacency = "\n".join(f"{node}: {', '.join(map(str, case['neighbors'][str(node)])) or '(none)'}" for node in case["node_order"])
        return f'''Task: Find a SHORTEST valid path from node {case['start']} to node {case['goal']} in this 32-node directed acyclic graph. Minimize the number of edges traversed. A valid path exists.

Rules: Nodes are numbered 0 through 31. Each move costs 1. "u: v1, v2" lists OUTGOING neighbors only: you may move from u to those nodes. Reverse movement is NOT allowed unless separately listed. "(none)" means no outgoing edges. Each node, including the start, may be visited at most once. There are no switches or gates. Node numbers and presentation order do not specify topological order. There is no additional step limit; any shortest path is accepted.

Outgoing neighbors:
{adjacency}

Plan the complete path before answering and verify direction, edge legality, and shortestness. The judge starts at the stated start and stops at the first illegal move. There is no intermediate feedback.

Output only valid JSON with a path array of objects containing integer from and to fields, and a final_node integer. Use one object per move. Do not include markdown or a reasoning transcript.'''

    def judge(self, case, output):
        route = output.get("path")
        if not isinstance(route, list):
            return {"pass": False, "execution_pass": False, "failure_type": "missing_path", "trace": []}
        node, seen, trace, illegal = case["start"], {case["start"]}, [], None
        for step, move in enumerate(route, 1):
            reason = None
            if not isinstance(move, dict) or type(move.get("from")) is not int or move["from"] != node:
                reason = "from_mismatch"
            elif type(move.get("to")) is not int or move["to"] not in case["neighbors"][str(node)]:
                reason = self.edge_error
            elif move["to"] in seen:
                reason = "repeated_node"
            if reason:
                illegal = {"step": step, "reason": reason, "move": move}
                break
            node = move["to"]
            seen.add(node)
            trace.append({"step": step, "node": node})
        reference = self.shortest(case)
        if reference is None:
            raise ValueError("Unsolvable case")
        execution = illegal is None and node == case["goal"]
        optimal = execution and len(route) == len(reference)
        reported_correct = type(output.get("final_node")) is int and output["final_node"] == node
        return {"pass": optimal, "execution_pass": execution, "legal": illegal is None, "optimal": optimal,
                "shortest_moves": len(reference), "attempted_moves": len(route), "executed_moves": len(trace),
                "final_node_actual": node, "reported_final_node_correct": reported_correct,
                "contract_pass": optimal and reported_correct,
                "failure_type": "illegal_move" if illegal else "goal_not_reached" if not execution else "non_shortest" if not optimal else None,
                "illegal_move": illegal, "trace": trace}


class UndirectedPathTask(DagTask):
    """Shortest-path judge for a case whose adjacency lists are symmetric."""

    condition = "path_undirected_256"
    edge_error = "nonexistent_undirected_edge"

    def prompt(self, case):
        node_count = case.get("node_count", len(case["neighbors"]))
        adjacency = "\n".join(
            f"{node}: {', '.join(map(str, case['neighbors'][str(node)])) or '(none)'}"
            for node in case["node_order"])
        return f'''Task: Find a SHORTEST valid path from node {case['start']} to node {case['goal']} in this {node_count}-node undirected graph. Minimize the number of edges traversed. A valid path exists.

Rules: Nodes are numbered 0 through {node_count - 1}. Each move costs 1. "u: v1, v2" lists every neighbor of u. Edges are bidirectional: if v is listed under u, the same edge is also listed under v, and you may traverse it in either direction. Each node, including the start, may be visited at most once. Node numbers and presentation order carry no spatial or numerical ordering. There is no additional step limit; any shortest path is accepted.

Neighbors:
{adjacency}

Plan the complete path before answering and verify edge legality and shortestness. The judge starts at the stated start and stops at the first illegal move. There is no intermediate feedback.

Output only valid JSON with a path array of objects containing integer from and to fields, and a final_node integer. Use one object per move. Do not include markdown or a reasoning transcript.'''


TASKS = {"blocks8": Blocks8Task(CONFIG["blocks"]), "blocks12": Blocks12Task(CONFIG["blocks"]),
         "path_dag": DagTask(), "path_undirected_256": UndirectedPathTask()}
