import gym
import numpy as np
import sys
from scipy import misc
from matplotlib import pyplot as plt

from gym import spaces


def up_scaler(grid, up_size):
    res = np.zeros(shape=np.asarray(np.shape(grid)) * up_size)
    for (x, y), value in np.ndenumerate(grid):
        res[x * up_size:x * up_size + up_size, y * up_size:y * up_size + up_size] = grid[x][y]
    return res


# noinspection PyChainedComparisons
class ArmEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    ON = 4
    OFF = 5

    MOVE_ACTIONS = {
        UP: [-1, 0],
        LEFT: [0, -1],
        DOWN: [1, 0],
        RIGHT: [0, 1],
    }

    def __init__(self, size_x, size_y, cubes_cnt):
        self._size_x = size_x
        self._size_y = size_y
        self._cubes_cnt = cubes_cnt
        # checking for grid overflow
        assert cubes_cnt < size_x * size_y, "Cubes overflow the grid"

        self.reset()

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Discrete(6)

    def ok(self, x, y):
        return 0 <= x < self._grid.shape[0] and 0 <= y < self._grid.shape[1]

    def ok_and_empty(self, x, y):
        return self.ok(x, y) and self._grid[x][y] == 0

    def grid_to_bin(self):
        grid = np.array(self._grid, copy=True)
        grid[self._arm_x, self._arm_y] = 2
        res = 0
        for x in np.nditer(grid):
            res = res * 10 + x
        res = res * 10 + self._magnet_toggle
        return res

    def _step(self, a):

        if a in self.MOVE_ACTIONS:
            cube_dx, cube_dy = self.MOVE_ACTIONS[self.DOWN]
            cube_x, cube_y = self._arm_x + cube_dx, self._arm_y + cube_dy
            if self._magnet_toggle and self.ok(cube_x, cube_y) and self._grid[cube_x][cube_y] == 1:
                new_arm_x, new_arm_y = self._arm_x + self.MOVE_ACTIONS[a][0], self._arm_y + self.MOVE_ACTIONS[a][1]
                new_cube_x, new_cube_y = new_arm_x + cube_dx, new_arm_y + cube_dy
                self._grid[cube_x][cube_y] = 0
                if self.ok_and_empty(new_arm_x, new_arm_y) and self.ok_and_empty(new_cube_x, new_cube_y):
                    self._arm_x, self._arm_y = new_arm_x, new_arm_y
                    self._grid[new_cube_x][new_cube_y] = 1
                else:
                    self._grid[cube_x][cube_y] = 1
            else:
                new_arm_x, new_arm_y = self._arm_x + self.MOVE_ACTIONS[a][0], self._arm_y + self.MOVE_ACTIONS[a][1]
                if self.ok_and_empty(new_arm_x, new_arm_y):
                    self._arm_x, self._arm_y = new_arm_x, new_arm_y
                else:
                    # cant move, mb -reward
                    pass
        elif a == self.ON:
            self._magnet_toggle = True
        elif a == self.OFF:

            cube_dx, cube_dy = self.MOVE_ACTIONS[self.DOWN]
            cube_x, cube_y = self._arm_x + cube_dx, self._arm_y + cube_dy
            if self.ok(cube_x, cube_y) and self._grid[cube_x, cube_y] == 1 and self._magnet_toggle:
                new_cube_x, new_cube_y = cube_x + cube_dx, cube_y + cube_dy
                while self.ok_and_empty(new_cube_x, new_cube_y):
                    new_cube_x, new_cube_y = new_cube_x + cube_dx, new_cube_y + cube_dy
                new_cube_x, new_cube_y = new_cube_x - cube_dx, new_cube_y - cube_dy
                self._grid[new_cube_x, new_cube_y], self._grid[cube_x, cube_y] = self._grid[cube_x, cube_y], self._grid[new_cube_x, new_cube_y]
                self._magnet_toggle = False

        observation = self.grid_to_bin()
        reward = -1
        info = None
        # self.render_to_image()
        # observation (object): agent's observation of the current environment
        # reward (float) : amount of reward returned after previous action
        # done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        # info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        height = self._grid.shape[0]
        for i in range(self._grid.shape[1]):
            if self._grid[height - 1][i] == 1 and self._grid[height - 2][i] == 1 and self._grid[height - 3][i] == 1:
                self._done = True
                reward = 100
                return observation, reward, self._done, info

        return observation, reward, self._done, info

    # return observation
    def _get_obs(self):
        pass

    # return: (states, observations)
    def _reset(self):
        self._grid = np.zeros(shape=(self._size_x, self._size_y), dtype=np.int32)
        self._arm_x = 0
        self._arm_y = 0
        self._done = False
        self._magnet_toggle = False

        cubes_left = self._cubes_cnt
        for (x, y), value in reversed(list(np.ndenumerate(self._grid))):
            if cubes_left == 0:
                break
            cubes_left -= 1
            self._grid[x, y] = 1

        exit(0)

        return self._get_obs()

    def _render(self, mode='human', close='false'):
        outfile = sys.stdout

        out = np.array(self._grid, copy=True)
        out[self._arm_x, self._arm_y] = 2

        outfile.write('\n')
        outfile.write(str(out))
        outfile.write('\n')

    def render_to_image(self):
        # Image size
        n_grid = np.array(self._grid, copy=True)

        n_grid[self._arm_x, self._arm_y] = 2

        n_grid = up_scaler(n_grid, 10)
        size_i = n_grid.shape[0]
        size_j = n_grid.shape[1]
        channels = 3

        # Create an empty image
        img = np.zeros((size_i, size_j, channels), dtype=np.uint8)
        # Set the RGB values
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if n_grid[x][y] == 2:
                    img[x][y] = (100, 120, 70)
                if n_grid[x][y] == 1:
                    img[x][y] = (230, 200, 150)
        # Display the image
        # misc.imshow(img)

        # Save the image
        misc.imsave("image.png", img)


if __name__ == "__main__":
    c = ArmEnv()
    from random_policy import random_policy

    plt.ion()
    env = ArmEnv()
    stats = random_policy(env, 3)
