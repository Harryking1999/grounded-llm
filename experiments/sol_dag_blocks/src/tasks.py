"""Study-specific environments and judges. No evaluator tools are sent to the model."""
from collections import deque
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
CONFIG = json.loads((STUDY / "configs/sol_medium.json").read_text())


class BlocksTask:
    def __init__(self, options):
        self.size = options["grid_size"]
        self.shapes = options["shape_rows"]
        self.subclasses = options["shape_subclasses"]
        self.placements = []
        self.by_action = {}
        for sid, shape in enumerate(self.shapes):
            rows = shape.split("/")
            offsets = [(r, c) for r, row in enumerate(rows) for c, x in enumerate(row) if x == "1"]
            for r in range(self.size - len(rows) + 1):
                for c in range(self.size - len(rows[0]) + 1):
                    tile = sum(1 << ((r + dr) * self.size + c + dc) for dr, dc in offsets)
                    action = {"shape_id": sid, "row": r, "col": c}
                    self.placements.append((tile, action))
                    self.by_action[sid, r, c] = tile

    def from_grid(self, grid):
        rows = grid.split("/") if isinstance(grid, str) else grid
        if (not isinstance(rows, list) or len(rows) != self.size
                or any(not isinstance(row, str) or len(row) != self.size or set(row) - {"0", "1"} for row in rows)):
            raise ValueError("Expected ten binary strings of length ten")
        return sum(1 << i for i, x in enumerate("".join(rows)) if x == "1")

    def to_rows(self, mask):
        return ["".join(str((mask >> (r * self.size + c)) & 1) for c in range(self.size)) for r in range(self.size)]

    def normalized_key(self, mask):
        coords = [divmod(i, self.size) for i in range(self.size ** 2) if mask >> i & 1]
        r0, c0 = min(r for r, _ in coords), min(c for _, c in coords)
        return tuple(sorted((r - r0, c - c0) for r, c in coords))

    def boundary(self, mask):
        out = 0
        for i in range(self.size ** 2):
            if mask >> i & 1:
                r, c = divmod(i, self.size)
                for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if 0 <= rr < self.size and 0 <= cc < self.size:
                        out |= 1 << (rr * self.size + cc)
        return out & ~mask

    def apply(self, mask, action, initial=None):
        if not isinstance(action, dict):
            raise ValueError("malformed_action")
        key = tuple(action.get(k) for k in ("shape_id", "row", "col"))
        if any(type(x) is not int for x in key):
            raise ValueError("noninteger_action")
        if key not in self.by_action:
            raise ValueError("unknown_shape_or_out_of_bounds")
        tile = self.by_action[key]
        if initial is not None and tile & initial != tile:
            raise ValueError("initial_empty_overlap")
        if tile & mask != tile:
            raise ValueError("already_removed_overlap")
        return mask ^ tile

    def prompt(self, case):
        shapes = ", ".join(f"{i}={s}" for i, s in enumerate(self.shapes))
        return f'''Task: Remove reusable shapes from this 10x10 binary grid until every cell is 0.

Rules: 1 means occupied and 0 means empty. An action is (shape_id, row, col), with zero-based row and column and the shape's top-left bounding-box anchor. Every 1-cell of the selected shape must overlap a CURRENT 1-cell on the board; only those cells become 0. You may NEVER remove an initially empty cell or a cell removed by an earlier action. All shape cells must be within the board. A shape's 0-cells are holes: they do not remove or constrain board cells. There is no gravity, inventory limit, or action-count limit. A complete decomposition exists. Any valid decomposition is accepted; you do not need to minimize actions or recover the original pieces. Only the fixed orientations listed below may be used. In the shape notation, / separates rows.
Shapes: {shapes}.

Initial grid rows, from row 0 through row 9 (each character is column 0 through column 9):
{chr(10).join(case['grid'].split('/'))}

Plan the COMPLETE sequence before answering. The judge executes it and stops at the first illegal action; there is NO intermediate environment feedback. For EVERY action, predict the entire board immediately AFTER that removal.

Output only valid JSON with an actions array. Each action object must contain integer shape_id, row, col, and board_after, an array of exactly ten strings of exactly ten binary characters (0 or 1), in row order. Also include final_status, "solved" or "unsolved", according to whether the final board is empty. Do not include markdown or a reasoning transcript.'''

    def judge(self, case, output):
        initial = mask = self.from_grid(case["grid"])
        actions = output.get("actions")
        if not isinstance(actions, list):
            return {"pass": False, "legal": False, "failure_type": "missing_actions", "trace": []}
        trace, illegal = [], None
        for step, action in enumerate(actions, 1):
            try:
                mask = self.apply(mask, action, initial)
            except ValueError as error:
                illegal = {"step": step, "reason": str(error), "action": action}
                break
            report = action.get("board_after")
            valid = isinstance(report, list)
            try:
                reported_mask = self.from_grid(report) if valid else None
            except ValueError:
                valid, reported_mask = False, None
            trace.append({"step": step, "action": action, "remaining_grid": "/".join(self.to_rows(mask)),
                          "remaining_cells": mask.bit_count(), "state_report_valid": valid,
                          "state_report_correct": valid and reported_mask == mask,
                          "cell_matches": self.size ** 2 - (reported_mask ^ mask).bit_count() if valid else 0})
        passed = illegal is None and mask == 0
        status_correct = output.get("final_status") == ("solved" if mask == 0 else "unsolved")
        return {"pass": passed, "execution_pass": passed, "legal": illegal is None, "solved": mask == 0,
                "failure_type": "illegal_action" if illegal else "stopped_early" if mask else None,
                "illegal_action": illegal, "attempted_actions": len(actions), "executed_actions": len(trace),
                "remaining_grid": "/".join(self.to_rows(mask)), "remaining_cells": mask.bit_count(),
                "state_reports_evaluated": len(trace), "state_reports_correct": sum(t["state_report_correct"] for t in trace),
                "state_reports_valid": sum(t["state_report_valid"] for t in trace),
                "reported_status_correct": status_correct,
                "contract_pass": passed and status_correct and all(t["state_report_correct"] for t in trace), "trace": trace}


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
