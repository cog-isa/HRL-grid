import itertools
import random
import sys

import numpy as np
from gym.envs.toy_text import discrete

# from environments.grid_maze_env.maze_world_env import categorical_sample
from environments.grid_maze_env.maze_world_env import MazeWorldEpisodeLength
from utils import plotting

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

MOVES = [UP, RIGHT, DOWN, LEFT]
MOVES_X_Y = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}


class MazeWorldTrain(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, maze, finish_reward=50, wall_minus_reward=-5, action_minus_reward=-1, episode_max_length=50):

        number_of_actions = 4

        P = {}

        state_id = 1
        maze_size_x = len(maze)
        maze_size_y = len(maze[0])
        state_id_table = np.zeros(shape=(maze_size_x, maze_size_y), dtype=np.int64)
        start_id = None

        self.possible_sets = []
        for i in range(2, maze_size_x - 2):
            for j in range(2, maze_size_y - 2):
                if maze[i, j] == 0:
                    self.possible_sets.append([i, j])

        start = random.choice(self.possible_sets)
        maze[start[0], start[1]] = 2

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
        self._episode_max_length = episode_max_length
        self._episode_length = 0
        self._state_id_table = state_id_table
        self._maze = maze

        super(MazeWorldTrain, self).__init__(max_state_id, number_of_actions, P, isd)

    def _step(self, a):
        transitions = self.P[self.s][a]
        i = MazeWorldEpisodeLength.categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        self._episode_length += 1
        if self._episode_length == self._episode_max_length:
            d = True
        assert (self._episode_length <= self._episode_max_length)
        return s, r, d, {"prob": p}

    def _reset(self):
        # clear previous start point
        for i in range(len(self._maze)):
            for j in range(len(self._maze[0])):
                if self._maze[i, j] == 2:
                    self._maze[i, j] = 0

        # randomly choose new start point
        start = random.choice(self.possible_sets)
        self.s = self._state_id_table[start[0]][start[1]]
        self._maze[start[0], start[1]] = 2
        # self.s = categorical_sample(self.isd, self.np_random)

        self.lastaction = None
        self._episode_length = 0
        return self.s

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


# randomly sets initial position
def place_start(maze):
    possible_sets = []
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i, j] == 0:
                possible_sets.append([i, j])

    start = random.choice(possible_sets)
    maze[start[0], start[1]] = 2
    return maze


# adds to the pattern exits and walls
def prepare_train_maze(maze):
    maze_size_x, maze_size_y = len(maze), len(maze[0])
    mz = np.ones(shape=(maze_size_x + 4, maze_size_y + 4))
    mz[2:maze_size_x + 2, 2:maze_size_y + 2] = maze
    for i in range(maze_size_x):
        if maze[i, 0] == 0 or maze[i, 0] == 2:
            mz[i + 2, 1] = 0
        if maze[i, maze_size_y - 1] == 0 or maze[i, maze_size_y - 1] == 2:
            mz[i + 2, maze_size_y + 2] = 0

    for j in range(maze_size_y):
        if maze[0, j] == 0 or maze[0, j] == 2:
            mz[1, j + 2] = 0
        if maze[maze_size_x - 1, j] == 0 or maze[maze_size_x - 1, j] == 2:
            mz[maze_size_x + 2, j + 2] = 0
    return mz


# place a finish depending on the direction
def place_finish(maze, direction):
    if direction == 0:
        for i in range(len(maze[0])):
            if maze[1, i] == 0:
                maze[1, i] = 3
    elif direction == 1:
        for i in range(len(maze)):
            if maze[i, len(maze[0]) - 2] == 0:
                maze[i, len(maze[0]) - 2] = 3
    elif direction == 2:
        for i in range(len(maze[0])):
            if maze[len(maze) - 2, i] == 0:
                maze[len(maze) - 2, i] = 3
    else:
        for i in range(len(maze)):
            if maze[i, 1] == 0:
                maze[i, 1] = 3
    return maze


class Option:
    def __init__(self, env, q_table=None):
        self.env = env
        if q_table:
            self.q_table = q_table
        else:
            self.q_table = np.zeros(shape=(env.observation_space.n - 1, env.action_space.n))

        self.state_no_to_index = {}
        state_no = 0
        n = 0
        for i in range(5):
            for j in range(5):
                if self.env._state_id_table[2 + i, 2 + j] != 0:
                    self.state_no_to_index[n] = state_no
                    state_no += 1
                n += 1

    def learning_option(self, num_episodes=400, eps=0.6, alpha=0.1, gamma=0.9, verbose=False):

        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

        for i_episode in range(num_episodes):

            if (i_episode + 1) % np.round(num_episodes / 10) == 0:
                eps = eps - 0.2 * eps

            state = self.env.reset()

            for t in itertools.count():

                # Take a step
                if np.random.rand(1) < eps:  # choose random action
                    action = np.random.choice(self.env.action_space.n, size=1)[0]
                else:
                    action = np.argmax(self.q_table[state - 1, :])

                next_state, reward, done, _ = self.env.step(action)

                # update q-table
                self.q_table[state - 1, action] = (1 - alpha) * self.q_table[state - 1, action] + alpha * (
                    reward + gamma * np.max(self.q_table[next_state - 1, :]))

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                if done:
                    break

                state = next_state

        # as we don't need to keep values for the states out of the pattern 5x5
        needed_indexes = np.sort(np.reshape(self.env._state_id_table[2:7, 2:7], (1, 25))[0])
        needed_indexes = needed_indexes[needed_indexes != 0] - 1
        self.q_table = self.q_table[needed_indexes]

        self.env.reset()

        if verbose:
            return stats, self.q_table
        else:
            return self
