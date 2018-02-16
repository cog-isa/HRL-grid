import itertools
import pickle
import sys
import time
import numpy as np
from Options_new.arm_env_options.baseline_q_table import test_policy
from Options_new.arm_env_options.opt_lift_cube_init import q_learning_opt
from environments.arm_env.arm_env_rand_init import ArmEnvRand
from utils import plotting


class ArmEnvOpt2(ArmEnvRand):

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
                self._grid[new_cube_x, new_cube_y], self._grid[cube_x, cube_y] = self._grid[cube_x, cube_y], self._grid[
                    new_cube_x, new_cube_y]
                self._magnet_toggle = False

        observation = self.grid_to_bin()
        self._current_state = observation
        reward = 5 * self._action_minus_reward
        if a == 0 or a == 2:
            reward += 50 * self._action_minus_reward
        info = None

        for i in range(self._grid.shape[1]):
            if self._grid[1, i] == 1 and self._grid[2, i] == 0:
                self._done = True
                reward += self._finish_reward
                info = True
                return observation, reward, self._done, info

        if self._episode_max_length <= self._episode_length:
            self._done = True
        return observation, reward, self._done, info

    def grid_to_bin2(self):
        grid = np.array(self.reset_grid, copy=True)
        s = []
        for i in np.nditer(grid):
            s.append(int(i))
        return tuple(s)


def main():

    q_table = {}
    initial_states = set()
    term_states = set()
    env_spec = set()

    n = 2000
    start_time = time.time()
    for i in range(n):
        env = ArmEnvOpt2(episode_max_length=100,
                 size_x=8,
                 size_y=4,
                 cubes_cnt=6,
                 action_minus_reward=-1,
                 finish_reward=200,
                 tower_target_size=6)

        if env.grid_to_bin2() in env_spec:
            continue
        else:
            env_spec.add(env.grid_to_bin2())

        stats, q_table, initial_states, term_states = q_learning_opt(env, 2000, q_table, initial_states, term_states, eps=0.1, alpha=0.7, gamma=1.0)
        print("\n n = ", i, "Len of init_st", len(initial_states), len(term_states), len(q_table), "\n")

    print("--- %s seconds ---" % (time.time() - start_time))
    print("\n Len of init_st", len(initial_states), len(term_states), len(q_table), "\n")

    with open('opt_q_table.txt', 'wb') as handle:
        pickle.dump(q_table, handle)

    with open('opt_init_st.txt', 'wb') as handle:
        pickle.dump(initial_states, handle)

    with open('opt_term_st.txt', 'wb') as handle:
        pickle.dump(term_states, handle)

    # print(q_table)

    env = ArmEnvOpt2(episode_max_length=100,
                 size_x=8,
                 size_y=4,
                 cubes_cnt=6,
                 action_minus_reward=-1,
                 finish_reward=200,
                 tower_target_size=6, seed=849)
    S, t = test_policy(env, q_table)

    env = ArmEnvOpt2(episode_max_length=100,
                 size_x=8,
                 size_y=4,
                 cubes_cnt=6,
                 action_minus_reward=-1,
                 finish_reward=200,
                 tower_target_size=6, seed=16374)
    S, t = test_policy(env, q_table)

    env = ArmEnvOpt2(episode_max_length=100,
                 size_x=8,
                 size_y=4,
                 cubes_cnt=6,
                 action_minus_reward=-1,
                 finish_reward=200,
                 tower_target_size=6, seed=2790)
    S, t = test_policy(env, q_table)

    env = ArmEnvOpt2(episode_max_length=100,
                 size_x=8,
                 size_y=4,
                 cubes_cnt=6,
                 action_minus_reward=-1,
                 finish_reward=200,
                 tower_target_size=6, seed=174)
    S, t = test_policy(env, q_table)

    env = ArmEnvOpt2(episode_max_length=100,
                 size_x=8,
                 size_y=4,
                 cubes_cnt=6,
                 action_minus_reward=-1,
                 finish_reward=200,
                 tower_target_size=6, seed=395)
    S, t = test_policy(env, q_table)

    env = ArmEnvOpt2(episode_max_length=100,
                 size_x=8,
                 size_y=4,
                 cubes_cnt=6,
                 action_minus_reward=-1,
                 finish_reward=200,
                 tower_target_size=6, seed=19476)
    S, t = test_policy(env, q_table)

    env = ArmEnvOpt2(episode_max_length=100,
                 size_x=8,
                 size_y=4,
                 cubes_cnt=6,
                 action_minus_reward=-1,
                 finish_reward=200,
                 tower_target_size=6, seed=4)
    S, t = test_policy(env, q_table)


if __name__ == '__main__':
    main()





