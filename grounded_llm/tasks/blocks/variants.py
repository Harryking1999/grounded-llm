"""Existing variable-size board rules used by text evaluation."""

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
