"""Complete standard-goal component, exact BFS to configured relation goals."""
import numpy as np
from experiments.task_text_baselines.src import puzzle8 as rules


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

    def features(self, ids):
        return np.eye(9, dtype=np.float32)[self.positions[np.asarray(ids)]].reshape(-1, 81)

    def baseline(self, source, goal):
        return (self.positions[np.asarray(source)] != self.positions[np.asarray(goal)]).sum(-1)

    def step(self, state, action):
        child = int(self.next[state, self.actions.index(action)])
        if child < 0:
            raise ValueError("Blank leaves board")
        return child
