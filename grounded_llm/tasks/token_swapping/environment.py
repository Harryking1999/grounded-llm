"""Unit-cost sequential adjacent swaps; complete labelled permutations only."""
import time


class TokenSwapping:
    def __init__(self, vertex_count, edges):
        if type(vertex_count) is not int or vertex_count < 2:
            raise ValueError('Require at least two vertices')
        self.n = vertex_count
        self.edges = tuple(sorted({tuple(sorted(edge)) for edge in edges}))
        if any(len(e) != 2 or e[0] == e[1] or any(type(v) is not int or not 0 <= v < self.n for v in e) for e in self.edges):
            raise ValueError('Invalid graph edge')
        self.edge_set = set(self.edges)
        self.goal = tuple(range(self.n))

    def apply(self, state, action):
        if (not isinstance(action, list) or len(action) != 2
                or any(type(v) is not int for v in action) or tuple(sorted(action)) not in self.edge_set):
            raise ValueError('Expected the two integer vertex IDs of an existing edge')
        a, b = action
        child = list(state)
        child[a], child[b] = child[b], child[a]
        return tuple(child), 1

    def solve(self, initial, max_states=1000000, timeout_seconds=30):
        """Exact bidirectional BFS; complete each depth layer before accepting a meet."""
        initial = tuple(initial)
        if sorted(initial) != list(self.goal):
            raise ValueError('Initial token placement must be a permutation')
        if initial == self.goal:
            return [], {'visited': 1}
        parents = [{initial: None}, {self.goal: None}]
        depths = [{initial: 0}, {self.goal: 0}]
        fronts = [{initial}, {self.goal}]
        started = time.monotonic()
        while all(fronts):
            side = 0 if len(fronts[0]) <= len(fronts[1]) else 1
            following, meetings = set(), []
            for state in sorted(fronts[side]):
                for a, b in self.edges:
                    child = list(state)
                    child[a], child[b] = child[b], child[a]
                    child = tuple(child)
                    if child in parents[side]:
                        continue
                    parents[side][child] = state, [a, b]
                    depths[side][child] = depths[side][state]+1
                    following.add(child)
                    if child in parents[1-side]:
                        meetings.append(child)
                if len(parents[0])+len(parents[1]) > max_states or time.monotonic()-started > timeout_seconds:
                    raise ValueError('Exact token-swapping search exceeded configured preparation bound')
            fronts[side] = following
            if meetings:
                meet = min(meetings, key=lambda s: (depths[0][s]+depths[1][s], s))
                prefix, suffix, state = [], [], meet
                while parents[0][state] is not None:
                    state, action = parents[0][state]
                    prefix.append(action)
                state = meet
                while parents[1][state] is not None:
                    state, action = parents[1][state]
                    suffix.append(action)
                return list(reversed(prefix))+suffix, {'visited': len(parents[0])+len(parents[1])}
        return None, {'visited': len(parents[0])+len(parents[1])}
