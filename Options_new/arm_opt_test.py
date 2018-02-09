import pickle
import itertools
from itertools import product
import sys
import random
import numpy as np
from gym import spaces

from environments.arm_env.arm_env import ArmEnv
from Options_new.arm_q_table import test_policy
from Options_new.arm_options import test_policy_short, test_policy_opt
from Options_new.arm_options2 import ArmEnvOpt2
from utils import plotting

class ArmEnv2(ArmEnv):

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

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = sys.stdout

        out = np.array(self._grid, copy=True)
        out[self._arm_x, self._arm_y] = 3 - self._magnet_toggle*1

        outfile.write('\n')
        outfile.write(str(out))
        outfile.write('\n')

    def step(self, a, options=False):

        if not options:
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

def q_learning_on_options(env, option_q_table, init_states, term_states, num_episodes, eps=0.5, alpha=0.5, gamma=1.0):
    moves_d = {0: 'LEFT', 1: "UP", 2: "RIGHT", 3: "DOWN", 4: "ON", 5: "OFF", 6: "OPTION"}
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
        # flag = 0
        if state not in q_table:
            q_table[state] = np.zeros(shape=n_actions+1*(state in init_states))


        for t in itertools.count():
            # WE CAN PRINT ENVIRONMENT STATE
            # if i_episode == num_episodes - 1:
            #    print("\n")
            #    print(state)
            #    env.render()

            # Take an action
            if np.random.rand(1) < eps:  # choose random option
                action0 = np.random.choice(np.arange(n_actions + 1*(state in init_states)), size=1)[0]
                #action0 = np.random.choice([0,2,4,5])
            else:
                action0 = np.argmax(q_table[state])
                # action0 = 0
                # for l in range(2, len(q_table[state])):
                #     if l != 3 and q_table[state][l] > q_table[state][action0]:
                #         action0 = l

            # if i_episode == num_episodes - 1: print(action0)
            # if action0 == 1 or action0 == 3:
            #     print("\n smth wrong")
            #     print(q_table[state][3])

            # if option is chosen
            if action0 >= n_actions:
                print("\n Option is chosen \n")
                # flag = 1
                # env.render()
                opt_rew = 0
                opt_t = 0
                opt_state = state

                while True:

                    action = np.argmax(option_q_table[opt_state])
                    opt_state, reward, done, _ = env.step(action, options=True)

                    opt_rew += reward
                    opt_t += 1

                    if done or opt_state in term_states:
                        break

                env._episode_length += 1

                # opt_rew += 5

                next_state = opt_state
                if next_state not in q_table:
                    q_table[next_state] = np.zeros(shape=n_actions+1*(next_state in init_states))

                q_table[state][action0] = (1 - alpha) * q_table[state][action0] + alpha * (
                    opt_rew + gamma ** opt_t * np.max(q_table[next_state]))

                # print(state, ":", q_table[state], "\n")

                # Update statistics
                stats.episode_rewards[i_episode] += opt_rew
                stats.episode_lengths[i_episode] = t

                if done:
                    break

                state = next_state

                # print("\n After option is executed \n")
                # env.render()

            else:
                next_state, reward, done, _ = env.step(action0)
                if next_state not in q_table:
                    q_table[next_state] = np.zeros(shape=n_actions+1*(next_state in init_states))

                q_table[state][action0] = (1 - alpha) * q_table[state][action0] + alpha * (
                    reward + gamma * np.max(q_table[next_state]))

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                if done:
                    break

                state = next_state
                # if flag:
                #     print("\n move after option", moves_d[action0])
                #     env.render()
                #     flag = 0

    return stats, q_table


def test_policy_opt(env, q_table, option_q_table, init_states, term_states):
    moves_d = {0: 'LEFT', 1: "UP", 2: "RIGHT", 3: "DOWN", 4: "ON", 5: "OFF", 6: "OPTION"}

    env.reset()
    state = env.get_current_state()
    n_actions = env.action_space.n
    S_r = 0
    S_t = 0
    print("\n Start of the episode")
    env.render()

    for t in itertools.count():
        # WE CAN PRINT ENVIRONMENT STATE

        # Take a step
        action = np.argmax(q_table[state])
        print(moves_d[action])  # env.get_tower_height()) #env.tower_height)

        # if option is chosen
        if action >= n_actions:
            opt_rew = 0
            opt_state = state

            while True:

                action0 = np.argmax(option_q_table[opt_state])
                opt_state, reward, done, _ = env.step(action0, options=True)

                opt_rew += reward

                if done or opt_state in term_states:
                    break

            next_state = opt_state
            reward = opt_rew
            env._episode_length += 1

        else:
            next_state, reward, done, _ = env.step(action)


        # Update statistics
        S_r += reward
        S_t = t

        if done:
            print("i was here")
            env.render()
            print("\n End of the episode")
            break

        state = next_state
        env.render()
    return S_r, S_t


# for i in term_states:
#          print("\n", np.array(i).reshape((5,3)), "\n")

def main():
    env = ArmEnv2(episode_max_length=50,
                     size_x=5,
                     size_y=3,
                     cubes_cnt=4,
                     action_minus_reward=-0.1,
                     finish_reward=100,
                     tower_target_size=4)

    with open('opt_q_table.txt', 'rb') as handle:
        opt_q_table = pickle.loads(handle.read())

    with open('opt_init_st.txt', 'rb') as handle:
        init_states = pickle.loads(handle.read())

    with open('opt_term_st.txt', 'rb') as handle:
        term_states = pickle.loads(handle.read())

    print(len(opt_q_table))
    print(opt_q_table)

    stats2, q_table2 = q_learning_on_options(env, opt_q_table, init_states, term_states, 10000)

    plotting.plot_episode_stats(stats2)
    #
    print(q_table2)
    print(len(q_table2))

    S, t = test_policy_opt(env, q_table2, opt_q_table, init_states, term_states)

    # env = ArmEnvOpt2(episode_max_length=50,
    #                  size_x=5,
    #                  size_y=3,
    #                  cubes_cnt=4,
    #                  action_minus_reward=-1,
    #                  finish_reward=100,
    #                  tower_target_size=4, seed=9)
    # S, t = test_policy(env, opt_q_table)
    #
    # env = ArmEnvOpt2(episode_max_length=50,
    #                  size_x=5,
    #                  size_y=3,
    #                  cubes_cnt=4,
    #                  action_minus_reward=-1,
    #                  finish_reward=100,
    #                  tower_target_size=4, seed=9)
    # S, t = test_policy(env, opt_q_table)
    #
    # env = ArmEnvOpt2(episode_max_length=50,
    #                  size_x=5,
    #                  size_y=3,
    #                  cubes_cnt=4,
    #                  action_minus_reward=-1,
    #                  finish_reward=100,
    #                  tower_target_size=4, seed=9)
    # S, t = test_policy(env, opt_q_table)
    #
    # env = ArmEnvOpt2(episode_max_length=50,
    #                  size_x=5,
    #                  size_y=3,
    #                  cubes_cnt=4,
    #                  action_minus_reward=-1,
    #                  finish_reward=100,
    #                  tower_target_size=4, seed=9)
    # S, t = test_policy(env, opt_q_table)
    #
    # env = ArmEnvOpt2(episode_max_length=50,
    #                  size_x=5,
    #                  size_y=3,
    #                  cubes_cnt=4,
    #                  action_minus_reward=-1,
    #                  finish_reward=100,
    #                  tower_target_size=4, seed=9)
    # S, t = test_policy(env, opt_q_table)
    #
    # env = ArmEnvOpt2(episode_max_length=50,
    #                  size_x=5,
    #                  size_y=3,
    #                  cubes_cnt=4,
    #                  action_minus_reward=-1,
    #                  finish_reward=100,
    #                  tower_target_size=4, seed=9)
    # S, t = test_policy(env, opt_q_table)


if __name__ == '__main__':
    main()