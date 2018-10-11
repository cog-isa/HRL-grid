import itertools
import operator
from time import sleep

import numpy as np
from collections import namedtuple, defaultdict

import sys
from gym.envs.toy_text import discrete
from tqdm import tqdm

from utils.plotting import plot_multi_test


def arg_max_action(q_dict, state, action_space):
    result_action = 0
    for action_to in range(action_space):
        if q_dict[state, action_to] > q_dict[state, result_action]:
            result_action = action_to
    return result_action


class TwoRooms(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    ACTIONS = namedtuple("ACTIONS", ["LEFT", "UP", "RIGHT", "DOWN", ])(
        UP=0,
        RIGHT=1,
        DOWN=2,
        LEFT=3,
    )

    CONSTANTS = namedtuple("CONSTANTS", ["FREE_CELL", "OBSTACLE", "TARGET"])(
        FREE_CELL=0,
        OBSTACLE=1,
        TARGET=2,
    )

    MOVES = [ACTIONS.UP, ACTIONS.RIGHT, ACTIONS.DOWN, ACTIONS.LEFT]
    MOVES_X_Y = {ACTIONS.UP: (0, -1), ACTIONS.RIGHT: (1, 0), ACTIONS.DOWN: (0, 1), ACTIONS.LEFT: (-1, 0)}

    def get_actions_as_dict(self):
        return {_: getattr(self.ACTIONS, _) for _ in self.ACTIONS._fields}

    def __init__(self):

        finish_reward = 1

        co = self.CONSTANTS

        self.code_middle = 2 ** 7

        action_mapping = {
            self.ACTIONS.RIGHT: np.array([0, 1]),
            self.ACTIONS.LEFT: np.array([0, -1]),
            self.ACTIONS.UP: np.array([-1, 0]),
            self.ACTIONS.DOWN: np.array([1, 0]),
        }

        self.mark = []

        # building 2-rooms maze
        self._maze = np.full(shape=(7, 12), fill_value=co.FREE_CELL).astype(np.int32)
        # feel boundaries of room with obstacles
        self._maze[0, :] = self._maze[:, 0] = co.OBSTACLE
        self._maze[self._maze.shape[0] - 1, :] = co.OBSTACLE
        self._maze[:, self._maze.shape[1] - 1] = co.OBSTACLE

        # separate rooms
        self._maze[:, self._maze.shape[1] // 2] = co.OBSTACLE

        # clear obstacles for door
        self._maze[self._maze.shape[0] // 2, self._maze.shape[1] // 2] = 0
        self._maze[self._maze.shape[0] // 2 - 1, self._maze.shape[1] // 2] = 0

        # placing target at the lower right corner of the right hand room
        self._maze[self._maze.shape[0] - 2, self._maze.shape[1] - 2] = co.TARGET

        prob = {}

        def append_transitions_from_cell(a_x, a_y, p):
            state = self.encode(a_x, a_y)
            p[state] = {a: [] for a in range(len(self.ACTIONS))}
            for a in self.ACTIONS:
                for a2 in self.ACTIONS:
                    dx, dy = action_mapping[a2]
                    a_n_x, a_n_y = a_x + dx, a_y + dy
                    if self._maze[a_n_x][a_n_y] == co.OBSTACLE:
                        new_state = state
                    else:
                        new_state = self.encode(a_n_x, a_n_y)
                    done = self._maze[a_n_x, a_n_y] == co.TARGET
                    reward = finish_reward if self._maze[a_n_x, a_n_y] == co.TARGET else -0.01
                    probability = 0.7 if a == a2 else 0.1
                    p[state][a].append((probability, new_state, reward, done))

        for agent_x1 in range(self._maze.shape[0]):
            for agent_y1 in range(self._maze.shape[1]):
                if self._maze[agent_x1][agent_y1] == co.OBSTACLE:
                    continue
                append_transitions_from_cell(agent_x1, agent_y1, prob)

        isd = []
        for x in range(1, self._maze.shape[0] - 1):
            for y in range(1, self._maze.shape[1] // 2):
                isd.append(self.encode(x, y))
        isd = np.array(isd)
        super(TwoRooms, self).__init__(self.encode(self._maze.shape[0] - 1, self._maze.shape[1] - 1), len(self.ACTIONS),
                                       prob, isd)

    def _reset(self):
        self.s = np.random.choice(self.isd, size=1)[0]
        self.lastaction = None
        return self.s

    def encode(self, x, y):
        # checking constraints for x,y coordinates
        assert 0 <= x < self._maze.shape[0] and 0 <= y < self._maze.shape[1]
        # checking constraints for shape[1]
        assert self._maze.shape[1] < self.code_middle

        return x * self.code_middle + y

    def decode(self, state):
        return state // self.code_middle, state % self.code_middle

    def _render(self, mode='human', close=False, mark=None):
        if close:
            return

        outfile = sys.stdout

        maze_size_x = len(self._maze)
        maze_size_y = len(self._maze[0])
        output = "\n"
        for i in range(maze_size_x):
            for j in range(maze_size_y):

                if self.s == self.encode(i, j):
                    output += " x "
                elif self.encode(i, j) in self.mark:
                    output += " b "
                else:
                    if self._maze[i][j] == 0:
                        output += " . "
                    if self._maze[i][j] == 1:
                        output += " O "
                    if self._maze[i][j] == 2:
                        output += " F "
                    if self._maze[i][j] == 3:
                        output += " F "
            output += '\n'
        outfile.write(output)


def q_learning(env, num_episodes, eps=0.1, alpha=0.1, gamma=0.9):
    to_plot = []

    q_table = defaultdict(lambda: 0)

    bottle_count = defaultdict(lambda: 0)
    bottle_value = defaultdict(lambda: 0)
    bottle_count[-1] = 0

    for _ in tqdm(range(num_episodes)):
        ep_reward = 0
        eps *= 0.999
        s = env.reset()

        while True:
            # sleep(0.05)
            if np.random.rand(1) < eps:
                action = np.random.choice(env.action_space.n, size=1)[0]
            else:
                action = arg_max_action(q_dict=q_table, state=s, action_space=env.action_space.n)

            next_s, reward, done, _ = env.step(action)
            # best = bottle_count.items()[0]
            # for i in bottle_count.items():
            #     pass
            max_key = max(bottle_count, key=lambda k: bottle_count[k])
            if eps < 0.05:
                print(max_key, bottle_count[max_key])
                env.mark = [max_key]
                env.render()
            # best = bottle_count.items(
            # for i in bottle_count:
            #     if

            bottle_count[next_s] += 1
            a = arg_max_action(q_dict=q_table, state=next_s, action_space=env.action_space.n)
            bottle_value[s] += alpha * (reward + gamma * q_table[next_s, a])

            q_table[s, action] = (1 - alpha) * q_table[s, action] + alpha * (reward + gamma * q_table[next_s, a])

            ep_reward += reward
            if done:
                break

            s = next_s
        to_plot.append(ep_reward)
    print(list(bottle_count))
    print(bottle_value)

    return to_plot, q_table


q_s, q_t = q_learning(TwoRooms(), 1000)
plot_multi_test([q_s, ])
