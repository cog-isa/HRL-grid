import itertools
from itertools import product
import sys
import random
import numpy as np
from gym import spaces

from environments.arm_env.arm_env import ArmEnv
from Options_new.arm_q_table import test_policy
from utils import plotting


class ArmEnvOpt2(ArmEnv):

    def place_cubes(self):
        self._grid = np.zeros(shape=(self._size_x, self._size_y), dtype=np.int32)

        cubes_left = self._cubes_cnt
        while cubes_left != 0:
            column = np.random.randint(self._size_y)
            for i in np.arange(self._size_x-1, 0, -1):
                if self._grid[i, column] == 0 and (self._size_x -i) < self._tower_target_size:
                    self._grid[i, column] = 1
                    cubes_left -= 1
                    break

    def __init__(self, size_x, size_y, cubes_cnt, episode_max_length, finish_reward, action_minus_reward, tower_target_size):
        self._size_x = size_x
        self._size_y = size_y
        self._cubes_cnt = cubes_cnt
        self._episode_max_length = episode_max_length
        self._finish_reward = finish_reward
        self._action_minus_reward = action_minus_reward
        self._tower_target_size = tower_target_size
        # checking for grid overflow
        assert cubes_cnt < size_x * size_y, "Cubes overflow the grid"
        self.place_cubes()
        self.reset_grid = np.copy(self._grid)

        self.reset()

        self.action_space = spaces.Discrete(6)
        self.grid_to_id = {}


    # return observation
    def _get_obs(self):
        pass

    def get_tower_height(self):
        h = 0
        for j in range(self._grid.shape[1]):
            t = 0
            for i in np.arange(self._grid.shape[0]-1, 0, -1):
                if self._grid[i, j] == 1 and self._grid[i-1, j] == 0:
                    if i+1 == self._grid.shape[0]:
                        t = self._grid.shape[0] - i
                        break
                    else:
                        if self._grid[i+1, j] == 1:
                            t = self._grid.shape[0] - i
                            break
            if t > h:
                h = t
        return h

    # return: (states, observations)
    def _reset(self):
        self._episode_length = 0
        self._done = False
        self._magnet_toggle = False
        self._current_state = 0
        self._grid = self.reset_grid
        self._arm_x = 0
        self._arm_y = np.random.randint(self._size_y)

        self.tower_height = self.get_tower_height()

        return self._get_obs()

    def _step(self, a):

        self._episode_length += 1

        if a in self.MOVE_ACTIONS:
            cube_dx, cube_dy = self.MOVE_ACTIONS[self.ACTIONS.DOWN]
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
        elif a == self.ACTIONS.ON:
            self._magnet_toggle = True
        elif a == self.ACTIONS.OFF:
            cube_dx, cube_dy = self.MOVE_ACTIONS[self.ACTIONS.DOWN]
            cube_x, cube_y = self._arm_x + cube_dx, self._arm_y + cube_dy
            if self.ok(cube_x, cube_y) and self._grid[cube_x, cube_y] == 1 and self._magnet_toggle:
                new_cube_x, new_cube_y = cube_x + cube_dx, cube_y + cube_dy
                while self.ok_and_empty(new_cube_x, new_cube_y):
                    new_cube_x, new_cube_y = new_cube_x + cube_dx, new_cube_y + cube_dy
                new_cube_x, new_cube_y = new_cube_x - cube_dx, new_cube_y - cube_dy
                self._grid[new_cube_x, new_cube_y], self._grid[cube_x, cube_y] = self._grid[cube_x, cube_y], self._grid[new_cube_x, new_cube_y]
                self._magnet_toggle = False

        observation = self.grid_to_bin()
        self._current_state = observation
        reward = self._action_minus_reward
        info = None
        # self.render_to_image()
        # observation (object): agent's observation of the current environment
        # reward (float) : amount of reward returned after previous action
        # done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        # info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        if self.get_tower_height() < self.tower_height:
                reward += 50 * self._action_minus_reward  # penalty for making the tower lower
        self.tower_height = self.get_tower_height()

        for i in range(self._grid.shape[1]):
            if self._grid[1, i] == 1 and self._grid[2, i] == 0:
                self._done = True
                reward += self._finish_reward
                return observation, reward, self._done, info

        if self._episode_max_length <= self._episode_length:
            self._done = True
        return observation, reward, self._done, info

    def _render(self, mode='human', close='false'):
        outfile = sys.stdout

        out = np.array(self._grid, copy=True)
        out[self._arm_x, self._arm_y] = 3 - self._magnet_toggle*1

        outfile.write('\n')
        outfile.write(str(out))
        outfile.write('\n')

    def _render(self, mode='human', close='false'):
        outfile = sys.stdout

        out = np.array(self._grid, copy=True)
        out[self._arm_x, self._arm_y] = 3 - self._magnet_toggle*1

        outfile.write('\n')
        outfile.write(str(out))
        outfile.write('\n')


def main():
    env = ArmEnvOpt2(episode_max_length=50,
                 size_x=5,
                 size_y=3,
                 cubes_cnt=4,
                 action_minus_reward=-1,
                 finish_reward=100,
                 tower_target_size=4)

if __name__ == '__main__':
    main()





