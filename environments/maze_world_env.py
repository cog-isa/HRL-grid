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

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

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
