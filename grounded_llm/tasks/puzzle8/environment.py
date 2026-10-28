"""Complete standard-goal component, exact BFS to configured relation goals."""
import numpy as np
import grounded_llm.tasks.puzzle8.rules as rules
class Puzzle8:
    name = "puzzle8"
    actions = tuple(rules.MOVES)
    diameter = 31
    feature_dim = 81
    goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    def __init__(self):
        states, index = [self.goal_state], {self.goal_state: 0}
        edges = []
        for state in states:
            row = [-1] * 4
            for action, child in rules.successors(state):
                if child not in index:
                    index[child] = len(states)
                    states.append(child)
                row[self.actions.index(action)] = index[child]
            edges.append(row)
        self.positions = np.asarray(states, dtype=np.int8)
        self.next = np.asarray(edges, dtype=np.int32)
        self.size = len(states)
        self.index = index
        self.goal_columns = {}
        self.tables = None

    def prepare_goals(self, goals):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
        rows, actions = np.where(self.next >= 0)
        graph = csr_matrix((np.ones(len(rows)), (rows, self.next[rows, actions])),
                           shape=(self.size, self.size))
        self.tables = shortest_path(graph, directed=False, unweighted=True,
                                    indices=goals).astype(np.int8)
        self.goal_columns = {int(g): i for i, g in enumerate(goals)}

    def distance(self, source, goal):
        if hasattr(self, "full_distances"):
            return self.full_distances[np.asarray(goal), np.asarray(source)].astype(np.int64)
        source, goal = np.broadcast_arrays(source, goal)
        columns = np.asarray([self.goal_columns[int(g)] for g in goal.ravel()]).reshape(goal.shape)
        return self.tables[columns, source].astype(np.int64)

    def successors(self, ids):
        return self.next[np.asarray(ids)]

    def validate(self, ids):
        values = np.asarray(ids)
        if values.dtype.kind not in 'iu' or np.any(values < 0) or np.any(values >= self.size):
            raise ValueError('State IDs must belong to the puzzle component')

    def features(self, ids):
        return np.eye(9, dtype=np.float32)[self.positions[np.asarray(ids)]].reshape(-1, 81)

    def goal_frame(self, source, goal):
        """Known graph symmetry: normalize goals within each blank-position orbit."""
        source, goal = np.broadcast_arrays(source, goal)
        self.validate(source); self.validate(goal)
        if not hasattr(self, '_goal_targets'):
            self._goal_targets = np.full(self.size, -1, dtype=np.int32)
            self._goal_gathers = np.empty((self.size, 9), dtype=np.int8)
            self._goal_labels = np.empty((self.size, 9), dtype=np.int8)
            self._goal_anchors = [0, self.step(0, 'Up'), self.step(self.step(0, 'Up'), 'Left')]
            square = np.arange(9).reshape(3, 3)
            self._board_frames = [np.rot90(square, k).ravel() for k in range(4)]
            self._board_frames += [np.fliplr(np.rot90(square, k)).ravel() for k in range(4)]
        flat_source, flat_goal = source.ravel(), goal.ravel()
        for g in np.unique(flat_goal):
            if self._goal_targets[g] < 0:
                board = self.positions[g]
                blank = int(np.argmin(board))
                kind = 0 if blank in (0, 2, 6, 8) else 2 if blank == 4 else 1
                anchor = self._goal_anchors[kind]
                target = self.positions[anchor]
                frame = next(t for t in self._board_frames if np.argmin(board[t]) == np.argmin(target))
                labels = np.empty(9, dtype=np.int8)
                labels[board[frame]] = target
                self._goal_gathers[g], self._goal_labels[g], self._goal_targets[g] = frame, labels, anchor
        boards = np.take_along_axis(self.positions[flat_source], self._goal_gathers[flat_goal], axis=1)
        boards = np.take_along_axis(self._goal_labels[flat_goal], boards, axis=1)
        normalized = np.asarray([self.index[tuple(board)] for board in boards], dtype=np.int64)
        targets = self._goal_targets[flat_goal]
        return normalized.reshape(source.shape), targets.reshape(goal.shape)

    def baseline(self, source, goal):
        return (self.positions[np.asarray(source)] != self.positions[np.asarray(goal)]).sum(-1)

    def step(self, state, action):
        child = int(self.next[state, self.actions.index(action)])
        if child < 0:
            raise ValueError("Blank leaves board")
        return child
