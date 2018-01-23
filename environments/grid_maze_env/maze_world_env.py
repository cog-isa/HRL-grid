from collections import namedtuple

import numpy as np

import sys
from gym.envs.toy_text import discrete

from environments.env_core import CoreEnv


class MazeWorldEpisodeLength(CoreEnv, discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    ACTIONS = namedtuple("ACTIONS", ["LEFT", "UP", "RIGHT", "DOWN", ])(
        UP=0,
        RIGHT=1,
        DOWN=2,
        LEFT=3,
    )

    MOVES = [ACTIONS.UP, ACTIONS.RIGHT, ACTIONS.DOWN, ACTIONS.LEFT]
    MOVES_X_Y = {ACTIONS.UP: (0, -1), ACTIONS.RIGHT: (1, 0), ACTIONS.DOWN: (0, 1), ACTIONS.LEFT: (-1, 0)}

    # noinspection PyUnresolvedReferences
    @staticmethod
    def categorical_sample(prob_n, np_random):
        """
        Sample from categorical distribution
        Each row specifies class probabilities
        """
        prob_n = np.asarray(prob_n)
        cs_prob_n = np.cumsum(prob_n)

        # cs_prob_n > np_random.rand() the same with:
        #  np.apply_along_axis(lambda x: x > np_random.rand(), 0, cs_prob_n)))
        return (cs_prob_n > np_random.rand()).argmax()

    def is_done(self):
        return self._is_done

    def get_actions_as_dict(self):
        return {_: getattr(self.ACTIONS, _) for _ in self.ACTIONS._fields}

    def get_current_state(self):
        return self._current_state

    def __init__(self, maze, finish_reward=100, wall_minus_reward=-5, action_minus_reward=-1, episode_max_length=100):
        number_of_actions = 4

        prob = {}

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

                prob[state_id] = {a: [] for a in range(number_of_actions)}

                for move in MazeWorldEpisodeLength.MOVES:
                    x, y = MazeWorldEpisodeLength.MOVES_X_Y[move]
                    x += i
                    y += j

                    new_state = state_id_table[x][y]

                    # if we are trying to go into the wall then ...
                    if maze[x][y] == 1:
                        reward = wall_minus_reward
                        new_state = state_id_table[i][j]
                    # if agents on the finish cell
                    elif maze[x][y] == 3:
                        reward = finish_reward
                        new_state = state_id_table[i][j]
                    # if agents on the start cell
                    elif maze[x][y] == 0 or maze[x][y] == 2:
                        reward = action_minus_reward
                    else:
                        raise ValueError
                    # [probability, state, reward, done]
                    prob[state_id][move] = [(1.0, new_state, reward, maze[x][y] == 3)]

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

        super(MazeWorldEpisodeLength, self).__init__(max_state_id, number_of_actions, prob, isd)

    def _step(self, a):
        transitions = self.P[self._current_state][a]
        i = self.categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self._current_state = s
        self._episode_length += 1
        if self._episode_length == self._episode_max_length or transitions[0][3]:
            self._is_done = True
        assert (self._episode_length <= self._episode_max_length)
        return s, r, self._is_done, {"prob": p}

    def _reset(self):
        self._current_state = self.categorical_sample(self.isd, self.np_random)
        self._episode_length = 0
        self._is_done = False
        return self._current_state

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = sys.stdout

        maze_size_x = len(self._maze)
        maze_size_y = len(self._maze[0])
        output = "\n"
        for i in range(maze_size_x):
            for j in range(maze_size_y):
                if self._current_state == self._state_id_table[i][j]:
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
                if self._current_state == self._state_id_table[i][j]:
                    return i, j


class MazeWorld(MazeWorldEpisodeLength):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, maze, finish_reward=100, wall_minus_reward=-5, action_minus_reward=-1):
        super(MazeWorld, self).__init__(maze=maze, finish_reward=finish_reward, wall_minus_reward=wall_minus_reward, action_minus_reward=action_minus_reward,
                                        episode_max_length=2 ** 128)
