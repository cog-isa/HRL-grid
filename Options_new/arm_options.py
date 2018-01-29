import itertools
from itertools import product
import sys
import random
import numpy as np

from environments.arm_env.arm_env import ArmEnv
from Options_new.arm_q_table import q_learning, test_policy
from utils import plotting


class ArmEnvOpt(ArmEnv):
    # return observation
    def _get_obs(self):
        pass

    # return: (states, observations)
    def _reset(self):
        self._episode_length = 0
        self._grid = np.zeros(shape=(self._size_x, self._size_y), dtype=np.int32)
        self._done = False
        self._magnet_toggle = False
        self._current_state = 0

        arm_pos = list(product(*[np.arange(self._size_x), np.arange(self._size_y)]))

        cubes_left = self._cubes_cnt
        for (x, y), value in reversed(list(np.ndenumerate(self._grid))):
            if cubes_left == 0:
                break
            cubes_left -= 1
            self._grid[x, y] = 1
            arm_pos.remove((x, y))

        arm = random.choice(arm_pos)
        self._arm_x = arm[0]
        self._arm_y = arm[1]

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

        for i in range(self._grid.shape[1]):
            if self._grid[1, i] == 1 and self._grid[2, i] == 0:
                self._done = True
                reward += self._finish_reward
                return observation, reward, self._done, info


        height = self._grid.shape[0]
        for i in range(self._grid.shape[1]):
            t = np.sum(self._grid[height - 1 - self._tower_target_size:height, i])
            if t == self._tower_target_size:
                self._done = True
                reward += self._finish_reward
                return observation, reward, self._done, info

        if self._episode_max_length <= self._episode_length:
            self._done = True
        return observation, reward, self._done, info

def q_learning_opt(env, num_episodes, eps=0.1, alpha=0.1, gamma=1.0):
    to_plot = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    q_table = {}
    term_states = set()

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.

        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
            eps = eps - 0.01 * eps

        # Reset the environment and pick the first action
        env.reset()
        state = env.get_current_state()
        if state not in q_table:
            q_table[state] = np.zeros(shape=env.action_space.n)

        for t in itertools.count():
            # WE CAN PRINT ENVIRONMENT STATE
            # env.render()

            # Take a step
            if np.random.rand(1) < eps:  # choose random action
                action = np.random.choice(env.action_space.n, size=1)[0]
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _ = env.step(action)
            if next_state not in q_table:
                q_table[next_state] = np.zeros(shape=env.action_space.n)
            q_table[state][action] = (1 - alpha) * q_table[state][action] + \
                                     alpha * (reward + gamma * np.max(q_table[next_state]))

            # Update statistics
            to_plot.episode_rewards[i_episode] += reward
            to_plot.episode_lengths[i_episode] = t

            if done:
                term_states.add(next_state)
                break

            state = next_state

    return to_plot, q_table, term_states


def q_learning_on_options(env, option_q_table, term_states, num_episodes, eps=0.1, alpha=0.1, gamma=1.0):
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    n_actions = env.action_space.n

    # initialize q-function
    q_table = {}

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.

        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
            eps = eps - 0.1 * eps

        # Reset the environment
        env.reset()
        state = env.get_current_state()
        if state not in q_table:
            q_table[state] = np.zeros(shape=n_actions+1*(state in option_q_table))

        for t in itertools.count():
            # WE CAN PRINT ENVIRONMENT STATE
            # if i_episode == num_episodes - 1:
            #    print("\n")
            #    print(state)
            #    env.render()

            # Take an action
            if np.random.rand(1) < eps:  # choose random option
                action0 = np.random.choice(np.arange(n_actions + 1*(state in option_q_table)), size=1)[0]
            else:
                action0 = np.argmax(q_table[state])

            # if i_episode == num_episodes - 1: print(action0)

            # if option is chosen
            if action0 >= n_actions:
                opt_rew = 0
                opt_t = 0
                opt_state = state

                while opt_t < 10:

                    action = np.argmax(option_q_table[opt_state])
                    opt_state, reward, done, _ = env.step(action)

                    opt_rew += reward
                    opt_t += 1

                    if done or opt_state in term_states:
                        break

                next_state = opt_state
                if next_state not in q_table:
                    q_table[next_state] = np.zeros(shape=n_actions+1*(next_state in option_q_table))

                q_table[state][action0] = (1 - alpha) * q_table[state][action0] + alpha * (
                    opt_rew + gamma ** opt_t * np.max(q_table[next_state]))

                # Update statistics
                stats.episode_rewards[i_episode] += opt_rew
                stats.episode_lengths[i_episode] = t

                if done:
                    break

                state = next_state

            else:
                next_state, reward, done, _ = env.step(action0)
                if next_state not in q_table:
                    q_table[next_state] = np.zeros(shape=n_actions+1*(next_state in option_q_table))
                q_table[state][action0] = (1 - alpha) * q_table[state][action0] + alpha * (
                    reward + gamma * np.max(q_table[next_state]))

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                if done:
                    break

                state = next_state

    return stats, q_table

def main():
    env = ArmEnvOpt(episode_max_length=500,
                 size_x=4,
                 size_y=3,
                 cubes_cnt=4,
                 action_minus_reward=-1,
                 finish_reward=100,
                 tower_target_size=3)
    stats, q_table, term = q_learning_opt(env, 2000)
    env = ArmEnv(episode_max_length=500,
                    size_x=4,
                    size_y=3,
                    cubes_cnt=4,
                    action_minus_reward=-1,
                    finish_reward=100,
                    tower_target_size=3)
    stats2, q_table2 = q_learning_on_options(env, q_table, term, 1000)

    #plotting.plot_episode_stats(stats)
    #for key in q_table:
    #    print(key, ":", q_table[key])

    #print("Testing policy 1")
    #S, t = test_policy(env, q_table)
    #print("Testing policy 2")
    #S, t = test_policy(env, q_table)


if __name__ == '__main__':
    main()
