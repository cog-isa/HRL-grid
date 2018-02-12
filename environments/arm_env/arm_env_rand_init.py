from itertools import product
import random
import numpy as np
from gym import spaces
from environments.arm_env.arm_env import ArmEnv

class ArmEnvRand(ArmEnv):

    def place_cubes(self, seed=None):
        if seed:
            np.random.seed(seed)
        self._grid = np.zeros(shape=(self._size_x, self._size_y), dtype=np.int32)

        cubes_left = self._cubes_cnt
        while cubes_left != 0:
            column = np.random.randint(self._size_y)
            for i in np.arange(self._size_x-1, 0, -1):
                if self._grid[i, column] == 0 and (self._size_x -i) < self._tower_target_size:
                    self._grid[i, column] = 1
                    cubes_left -= 1
                    break

    def __init__(self, size_x, size_y, cubes_cnt, episode_max_length, finish_reward, action_minus_reward, tower_target_size, seed = None):
        self._size_x = size_x
        self._size_y = size_y
        self._cubes_cnt = cubes_cnt
        self._episode_max_length = episode_max_length
        self._finish_reward = finish_reward
        self._action_minus_reward = action_minus_reward
        self._tower_target_size = tower_target_size
        # checking for grid overflow
        assert cubes_cnt < size_x * size_y, "Cubes overflow the grid"

        self.place_cubes(seed)
        self.reset_grid = np.copy(self._grid)

        self.reset()

        self.action_space = spaces.Discrete(6)
        self.grid_to_id = {}

    def _reset(self):
        self._episode_length = 0
        self._done = False
        self._magnet_toggle = False
        self._grid = np.copy(self.reset_grid)

        # cartesian product
        arm_pos = list(product(*[np.arange(self._size_x), np.arange(self._size_y)]))

        for (x, y), value in reversed(list(np.ndenumerate(self._grid))):
            if self._grid[x, y] == 1:
                arm_pos.remove((x, y))

        arm = random.choice(arm_pos)
        self._arm_x = arm[0]
        self._arm_y = arm[1]

        self._current_state = self.grid_to_bin()
        self.tower_height = self.get_tower_height()

        return self._get_obs()