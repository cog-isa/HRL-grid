import numpy as np

import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

MOVES = [UP, RIGHT, DOWN, LEFT]
MOVES_X_Y = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}


class MazeWorld(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, maze, finish_reward=100, wall_minus_reward=-5, action_minus_reward=-1):

        number_of_actions = 4

        P = {}

        state_id = 1
        maze_size_x = len(maze)
        maze_size_y = len(maze[0])
        state_id_table = np.zeros(shape=(maze_size_x, maze_size_y), dtype=np.int64)
        start_id = None

        for i in range(maze_size_x):
            for j in range(maze_size_y):
                if maze[i][j] == 1:
                    continue

                state_id_table[i][j] = state_id
                if maze[i][j] == 2:
                    start_id = state_id_table[i][j]
                state_id += 1
        max_state_id = state_id
        for i in range(maze_size_x):
            for j in range(maze_size_y):
                state_id = state_id_table[i][j]
                if maze[i][j] == 1:
                    continue

                P[state_id] = {a: [] for a in range(number_of_actions)}

                for move in MOVES:
                    x, y = MOVES_X_Y[move]
                    x += i
                    y += j

                    new_state = state_id_table[x][y]

                    # if we try to go to the wall
                    reward = 0
                    if maze[x][y] == 1:
                        reward = wall_minus_reward
                        new_state = state_id_table[i][j]
                    # if it is the terminal state
                    elif maze[x][y] == 3:
                        reward = finish_reward
                        new_state = state_id_table[i][j]
                    # if the starting or empty cell
                    elif maze[x][y] == 0 or maze[x][y] == 2:
                        reward = action_minus_reward
                    else:
                        raise ValueError
                    P[state_id][move] = [(1.0, new_state, reward, maze[x][y] == 3)]

        isd = np.zeros(max_state_id)
        isd[start_id] = 1.0

        # uncomment for on-model
        # self.P = P

        # will only be used for our own render method
        self._state_id_table = state_id_table
        self._maze = maze

        super(MazeWorld, self).__init__(max_state_id, number_of_actions, P, isd)

    def is_done(self):
        return self._done

    def get_current_info(self):
        return None

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = sys.stdout

        maze_size_x = len(self._maze)
        maze_size_y = len(self._maze[0])
        output = "\n"
        for i in range(maze_size_x):
            for j in range(maze_size_y):
                if self.s == self._state_id_table[i][j]:
                    output += " x "
                else:
                    if self._maze[i][j] == 0:
                        output += " . "
                    if self._maze[i][j] == 1:
                        output += " O "
                    if self._maze[i][j] == 2:
                        output += " S "
                    if self._maze[i][j] == 3:
                        output += " F "
            output += '\n'
        outfile.write(output)


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class MazeWorldEpisodeLength(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, maze, finish_reward=100, wall_minus_reward=-5, action_minus_reward=-1, episode_max_length=100):

        number_of_actions = 4

        P = {}

        state_id = 1
        maze_size_x = len(maze)
        maze_size_y = len(maze[0])
        state_id_table = np.zeros(shape=(maze_size_x, maze_size_y), dtype=np.int64)
        start_id = None

        for i in range(maze_size_x):
            for j in range(maze_size_y):
                if maze[i][j] == 1:
                    continue

                state_id_table[i][j] = state_id
                if maze[i][j] == 2:
                    start_id = state_id_table[i][j]
                state_id += 1
        max_state_id = state_id
        for i in range(maze_size_x):
            for j in range(maze_size_y):
                state_id = state_id_table[i][j]
                if maze[i][j] == 1:
                    continue

                P[state_id] = {a: [] for a in range(number_of_actions)}

                for move in MOVES:
                    x, y = MOVES_X_Y[move]
                    x += i
                    y += j

                    new_state = state_id_table[x][y]

                    # if we try to go to the wall
                    reward = 0
                    if maze[x][y] == 1:
                        reward = wall_minus_reward
                        new_state = state_id_table[i][j]
                    # if it is the terminal state
                    elif maze[x][y] == 3:
                        reward = finish_reward
                        new_state = state_id_table[i][j]
                    # if the starting or empty cell
                    elif maze[x][y] == 0 or maze[x][y] == 2:
                        reward = action_minus_reward
                    else:
                        raise ValueError
                    P[state_id][move] = [(1.0, new_state, reward, maze[x][y] == 3)]

        isd = np.zeros(max_state_id)
        isd[start_id] = 1.0

        # uncomment for on-model
        # self.P = P

        self._is_done = False

        # will only be used for our own render method
        self._episode_max_length = episode_max_length
        self._episode_length = 0
        self._state_id_table = state_id_table
        self._maze = maze

        n = 0
        k = 0
        self.state_to_state_no = {}
        self.state_to_pattern_no = {}
        for i in range(int(len(self._state_id_table) / 5)):
            for j in range(int(len(self._state_id_table[0]) / 5)):
                for q in range(5):
                    for z in range(5):
                        if self._state_id_table[5 * i + q, 5 * j + z] != 0:
                            self.state_to_state_no[self._state_id_table[5 * i + q, 5 * j + z]] = n
                            self.state_to_pattern_no[self._state_id_table[5 * i + q, 5 * j + z]] = k
                        n += 1
                k += 1
                n = 0

        super(MazeWorldEpisodeLength, self).__init__(max_state_id, number_of_actions, P, isd)

    def _step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        self._episode_length += 1
        if self._episode_length == self._episode_max_length:
            self._is_done = True
        assert (self._episode_length <= self._episode_max_length)
        return s, r, self._is_done, {"prob": p}

    def _reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        self._episode_length = 0
        self._is_done = True
        return self.s

    def get_current_info(self):
        return None

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = sys.stdout

        maze_size_x = len(self._maze)
        maze_size_y = len(self._maze[0])
        output = "\n"
        for i in range(maze_size_x):
            for j in range(maze_size_y):
                if self.s == self._state_id_table[i][j]:
                    output += " x "
                else:
                    if self._maze[i][j] == 0:
                        output += " . "
                    if self._maze[i][j] == 1:
                        output += " O "
                    if self._maze[i][j] == 2:
                        output += " S "
                    if self._maze[i][j] == 3:
                        output += " F "
            output += '\n'
        outfile.write(output)

    def get_agent_x_y(self):
        maze_size_x = len(self._maze)
        maze_size_y = len(self._maze[0])

        for i in range(maze_size_x):
            for j in range(maze_size_y):
                if self.s == self._state_id_table[i][j]:
                    return i, j
