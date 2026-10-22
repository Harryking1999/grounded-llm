"""Fogleman six-by-six Rush Hour; a slide of any legal length costs one move."""
from collections import deque


class RushHour:
    def __init__(self, board):
        if len(board) != 36 or any(c not in 'ox.ABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in board):
            raise ValueError('Expected a 36-cell Rush Hour board')
        self.walls = sum(1 << i for i, c in enumerate(board) if c == 'x')
        self.pieces, starts, self.masks = [], [], []
        for label in sorted(set(board) - {'x', 'o', '.'}):
            cells = [i for i, c in enumerate(board) if c == label]
            horizontal = len({i // 6 for i in cells}) == 1
            if len(cells) not in (2, 3) or cells != [cells[0]+k*(1 if horizontal else 6) for k in range(len(cells))]:
                raise ValueError('Each vehicle must be a contiguous length-two or length-three segment')
            lane = cells[0] // 6 if horizontal else cells[0] % 6
            self.pieces.append((label, horizontal, len(cells), lane))
            starts.append(cells[0] % 6 if horizontal else cells[0] // 6)
            self.masks.append(tuple(sum(1 << (lane*6+p+k if horizontal else (p+k)*6+lane)
                                       for k in range(len(cells))) for p in range(7-len(cells))))
        if not self.pieces or self.pieces[0][:3] != ('A', True, 2):
            raise ValueError('A must be the horizontal length-two target car')
        self.labels = {piece[0]: i for i, piece in enumerate(self.pieces)}
        self.start = tuple(starts)

    def occupancy(self, state):
        occupied = self.walls
        for i, position in enumerate(state):
            occupied |= self.masks[i][position]
        return occupied

    def transitions(self, state):
        occupied = self.occupancy(state)
        for i, (label, _, size, _) in enumerate(self.pieces):
            other = occupied ^ self.masks[i][state[i]]
            for direction in (-1, 1):
                target = state[i] + direction
                while 0 <= target <= 6-size and not self.masks[i][target] & other:
                    child = list(state)
                    child[i] = target
                    yield {'vehicle': label, 'displacement': target-state[i]}, tuple(child), 1
                    target += direction

    def apply(self, state, action):
        if (not isinstance(action, dict) or set(action) != {'vehicle', 'displacement'}
                or not isinstance(action['vehicle'], str) or action['vehicle'] not in self.labels
                or type(action['displacement']) is not int or action['displacement'] == 0):
            raise ValueError('Expected a vehicle label and a nonzero integer displacement')
        i = self.labels[action['vehicle']]
        target = state[i] + action['displacement']
        if not 0 <= target < len(self.masks[i]):
            raise ValueError('Vehicle would leave the board')
        other = self.occupancy(state) ^ self.masks[i][state[i]]
        direction = 1 if target > state[i] else -1
        for position in range(state[i]+direction, target+direction, direction):
            if self.masks[i][position] & other:
                raise ValueError('Vehicle would pass through another vehicle or wall')
        child = list(state)
        child[i] = target
        return tuple(child), 1

    @staticmethod
    def terminal(state):
        return state[0] == 4

    def render(self, state):
        board = ['x' if self.walls >> i & 1 else '.' for i in range(36)]
        for i, (label, _, _, _) in enumerate(self.pieces):
            for cell in range(36):
                if self.masks[i][state[i]] >> cell & 1:
                    board[cell] = label
        return [''.join(board[i:i+6]) for i in range(0, 36, 6)]

    def solve(self, max_states=100000):
        """Enumerate the component and return a shortest exit plan and its exact size."""
        parents = {self.start: None}
        queue, first_goal = deque([self.start]), None
        while queue:
            state = queue.popleft()
            if self.terminal(state) and first_goal is None:
                first_goal = state
            for action, child, _ in self.transitions(state):
                if child not in parents:
                    if len(parents) >= max_states:
                        raise ValueError('Reachable component exceeds the configured bound')
                    parents[child] = state, action
                    queue.append(child)
        if first_goal is None:
            return None, len(parents)
        actions, state = [], first_goal
        while parents[state] is not None:
            state, action = parents[state]
            actions.append(action)
        return list(reversed(actions)), len(parents)
