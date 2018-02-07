import itertools
from itertools import product
import sys
import random
import numpy as np
from gym import spaces
import pickle

from environments.arm_env.arm_env import ArmEnv
from Options_new.arm_q_table import test_policy
from Options_new.arm_options import test_policy_short, test_policy_opt
from utils import plotting


class ArmEnvOpt2(ArmEnv):

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


    # return observation
    def _get_obs(self):
        pass

    def get_tower_height(self):
        h = 0
        for j in range(self._grid.shape[1]):
            t = 0
            for i in np.arange(self._grid.shape[0]-1, 0, -1):
                if self._grid[i, j] == 1 and self._grid[i-1, j] == 0 and (i+1 == self._grid.shape[0] or self._grid[i+1, j] == 1):
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
        self._grid = np.copy(self.reset_grid)

        # cartesian product
        arm_pos = list(product(*[np.arange(self._size_x), np.arange(self._size_y)]))

        for (x, y), value in reversed(list(np.ndenumerate(self._grid))):
            if self._grid[x, y] == 1:
                arm_pos.remove((x, y))

        arm = random.choice(arm_pos)
        self._arm_x = arm[0]
        self._arm_y = arm[1]
        # self._arm_x = 0
        # self._arm_y = np.random.randint(self._size_y)

        self._current_state = self.grid_to_bin()
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
                self._grid[new_cube_x, new_cube_y], self._grid[cube_x, cube_y] = self._grid[cube_x, cube_y], self._grid[
                    new_cube_x, new_cube_y]
                self._magnet_toggle = False

        observation = self.grid_to_bin()
        self._current_state = observation
        reward = 5*self._action_minus_reward
        if a == 0 or a == 2:
            reward += 50*self._action_minus_reward
        info = None
        # self.render_to_image()
        # observation (object): agent's observation of the current environment
        # reward (float) : amount of reward returned after previous action
        # done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        # info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        # if self.get_tower_height() < self.tower_height:
        #         reward += 50 * self._action_minus_reward  # penalty for making the tower lower
        # self.tower_height = self.get_tower_height()

        for i in range(self._grid.shape[1]):
            if self._grid[1, i] == 1 and self._grid[2, i] == 0:
                #if ( (self._grid[2:] == self.initial_grid[2:]).size - np.sum((self._grid[2:] == self.initial_grid[2:])) ) == 1:
                self._done = True
                reward += self._finish_reward
                info = True
                return observation, reward, self._done, info

        if self._episode_max_length <= self._episode_length:
            self._done = True
        return observation, reward, self._done, info

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = sys.stdout

        out = np.array(self._grid, copy=True)
        out[self._arm_x, self._arm_y] = 3 - self._magnet_toggle*1

        outfile.write('\n')
        outfile.write(str(out))
        outfile.write('\n')


def q_learning_opt(env, num_episodes, q_table, initial_states, term_states, eps=0.5, alpha=0.5, gamma=1.0):
    to_plot = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.

        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
            eps = eps - 0.1 * eps

        # Reset the environment
        env.reset()
        state = env.get_current_state()
        initial_states.add(state)

        if state not in q_table:
            q_table[state] = np.zeros(shape=env.action_space.n)

        for t in itertools.count():

            # Take a step
            if np.random.rand(1) < eps:  # choose random action
                action = np.random.choice(env.action_space.n, size=1)[0]
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, info = env.step(action)
            if next_state not in q_table:
                q_table[next_state] = np.zeros(shape=env.action_space.n)

            q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * (reward + gamma * np.max(q_table[next_state]))

            # Update statistics
            to_plot.episode_rewards[i_episode] += reward
            to_plot.episode_lengths[i_episode] = t

            if done:
                if info:
                    term_states.add(next_state)
                break

            state = next_state

    return to_plot, q_table, initial_states, term_states


def main():
    q_table = {}
    initial_states = set()
    term_states = set()
    n = 100
    for i in range(n):
        env = ArmEnvOpt2(episode_max_length=50,
                 size_x=5,
                 size_y=3,
                 cubes_cnt=4,
                 action_minus_reward=-1,
                 finish_reward=100,
                 tower_target_size=4)
        stats, q_table, initial_states, term_states = q_learning_opt(env, 2000, q_table, initial_states, term_states)
        print("\n n = ", i, "Len of init_st", len(initial_states), len(term_states), len(q_table), "\n")

    print("\n Len of init_st", len(initial_states), len(term_states), len(q_table), "\n")

    with open('opt_q_table.txt', 'wb') as handle:
        pickle.dump(q_table, handle)

    with open('opt_init_st.txt', 'wb') as handle:
        pickle.dump(initial_states, handle)

    with open('opt_term_st.txt', 'wb') as handle:
        pickle.dump(term_states, handle)

    env = ArmEnvOpt2(episode_max_length=50,
                 size_x=5,
                 size_y=3,
                 cubes_cnt=4,
                 action_minus_reward=-1,
                 finish_reward=100,
                 tower_target_size=4, seed=345)
    S, t = test_policy(env, q_table)

    env = ArmEnvOpt2(episode_max_length=50,
                 size_x=5,
                 size_y=3,
                 cubes_cnt=4,
                 action_minus_reward=-1,
                 finish_reward=100,
                 tower_target_size=4, seed=345)
    S, t = test_policy(env, q_table)

    env = ArmEnvOpt2(episode_max_length=50,
                 size_x=5,
                 size_y=3,
                 cubes_cnt=4,
                 action_minus_reward=-1,
                 finish_reward=100,
                 tower_target_size=4, seed=345)
    S, t = test_policy(env, q_table)

    env = ArmEnvOpt2(episode_max_length=50,
                     size_x=5,
                     size_y=3,
                     cubes_cnt=4,
                     action_minus_reward=-1,
                     finish_reward=100,
                     tower_target_size=4, seed=345)
    S, t = test_policy(env, q_table)


if __name__ == '__main__':
    main()





