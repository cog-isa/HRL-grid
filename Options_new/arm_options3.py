import itertools
from itertools import product
import sys
import random
import numpy as np

from environments.arm_env.arm_env import ArmEnv
from Options_new.arm_q_table import q_learning, test_policy
from utils import plotting


class ArmEnvOpt(ArmEnv):

    # return: (states, observations)
    def _reset(self):
        self._episode_length = 0
        self._grid = np.zeros(shape=(self._size_x, self._size_y), dtype=np.int32)
        self._done = False
        self._magnet_toggle = False

        # cartesian product
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
        # self._arm_x = 0
        # self._arm_y = np.random.randint(self._size_y)

        self._current_state = self.grid_to_bin()
        self.initial_grid = np.copy(self._grid)

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
        reward = self._action_minus_reward
        info = None
        # self.render_to_image()
        # observation (object): agent's observation of the current environment
        # reward (float) : amount of reward returned after previous action
        # done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        # info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

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

    def _render(self, mode='human', close='false'):
        outfile = sys.stdout

        out = np.array(self._grid, copy=True)
        out[self._arm_x, self._arm_y] = 3 - self._magnet_toggle*1

        outfile.write('\n')
        outfile.write(str(out))
        outfile.write('\n')


def q_learning_opt(env, num_episodes, eps=0.6, alpha=0.7, gamma=1.0):
    to_plot = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    q_table = {}
    initial_states = set()
    term_states = set()

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
            q_table[state][action] = (1 - alpha) * q_table[state][action] + \
                                     alpha * (reward + gamma * np.max(q_table[next_state]))

            # Update statistics
            to_plot.episode_rewards[i_episode] += reward
            to_plot.episode_lengths[i_episode] = t

            if done and info:
                term_states.add(next_state)
                break

            state = next_state

    return to_plot, q_table, initial_states, term_states


def q_learning_on_options(env, option_q_table, init_states, term_states, num_episodes, eps=0.5, alpha=0.5, gamma=1.0):
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
            else:
                action0 = np.argmax(q_table[state])

            # if i_episode == num_episodes - 1: print(action0)

            # if option is chosen
            if action0 >= n_actions:
                # print("\n Yahhooo \n")
                opt_rew = 0
                opt_t = 0
                opt_state = state

                while True:

                    action = np.argmax(option_q_table[opt_state])
                    opt_state, reward, done, _ = env.step(action)

                    opt_rew += reward
                    opt_t += 1

                    if done or opt_state in term_states:
                        break

                next_state = opt_state
                if next_state not in q_table:
                    q_table[next_state] = np.zeros(shape=n_actions+1*(next_state in init_states))

                q_table[state][action0] = (1 - alpha) * q_table[state][action0] + alpha * (
                    opt_rew + gamma ** opt_t * np.max(q_table[next_state]))

                print(state, ":", q_table[state], "\n")

                # Update statistics
                stats.episode_rewards[i_episode] += opt_rew
                stats.episode_lengths[i_episode] = t

                if done:
                    break

                state = next_state

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

        # if option is chosen
        if action >= n_actions:
            opt_rew = 0
            opt_t = 0
            opt_state = state

            while True:

                action = np.argmax(option_q_table[opt_state])
                opt_state, reward, done, _ = env.step(action)

                opt_rew += reward
                opt_t += 1

                if done or opt_state in term_states:
                    break

            next_state = opt_state

            if done:
                break

            state = next_state


        else:
            next_state, reward, done, _ = env.step(action)

            if done:
                break

            state = next_state

        print(moves_d[action]) # env.get_tower_height()) #env.tower_height)

        # Update statistics
        S_r += reward
        S_t = t

        if done:
            env.render()
            print("\n End of the episode")
            break

        state = next_state
        env.render()
    return S_r, S_t

def test_policy_short(env, q_table):
    moves_d = {0: 'LEFT', 1: "UP", 2: "RIGHT", 3: "DOWN", 4: "ON", 5: "OFF", 6: "option"}
    env.reset()
    state = env.get_current_state()
    S_r = 0
    S_t = 0
    print("\n Start of the episode")
    env.render()
    #print("Tower height:", env.get_tower_height())

    for t in itertools.count():
        # WE CAN PRINT ENVIRONMENT STATE

        # Take a step
        action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)

        # Update statistics
        S_r += reward
        S_t = t

        if done:
            env.render()
            #print("Tower height:", env.get_tower_height())
            print("\nEnd of the episode")
            break

        state = next_state
    return S_r, S_t


def main():
    env = ArmEnvOpt(episode_max_length=100,
                 size_x=5,
                 size_y=5,
                 cubes_cnt=6,
                 action_minus_reward=-1,
                 finish_reward=200,
                 tower_target_size=4)
    stats, q_table, init_st, term_st = q_learning_opt(env, 3000)
    print("\n Len of init_st", len(init_st), len(term_st), len(q_table), "\n")
    # for i in term_st:
    #     print("\n", np.array(i).reshape((5,5)), "\n")
    env2 = ArmEnv(episode_max_length=100,
                 size_x=5,
                 size_y=5,
                 cubes_cnt=6,
                 action_minus_reward=-1,
                 finish_reward=100,
                 tower_target_size=4)
    stats2, q_table2 = q_learning_on_options(env2, q_table, init_st, term_st, 10000)

    S, t = test_policy_opt(env2, q_table2, q_table, init_st, term_st)
    # stats3, q_table3 = q_learning(env, 5000)

    # plotting.plot_multi_test(smoothing_window=30,
    #                          x_label="episode",
    #                          y_label="smoothed rewards",
    #                          curve_to_draw=[stats2.episode_rewards,
    #                                         stats3.episode_rewards],
    #                          labels=["options", "q-learning"]
    #                          )

    plotting.plot_episode_stats(stats2)
    print(q_table2)
    #for key in q_table:
    #    print(key, ":", q_table[key])

    # print("\nTesting policy 1")
    # S, t = test_policy_short(env, q_table)
    # print("Testing policy 2")
    # S, t = test_policy_short(env, q_table)
    # print("Testing policy 3")
    # S, t = test_policy_short(env, q_table)
    # print("Testing policy 4")
    # S, t = test_policy_short(env, q_table)
    # print("Testing policy 5")
    # S, t = test_policy_short(env, q_table)
    # print("Testing policy 6")
    # S, t = test_policy_short(env, q_table)
    # print("Testing policy 7")
    # S, t = test_policy_short(env, q_table)
    # print("Testing policy 8")
    # S, t = test_policy_short(env, q_table)
    # print("Testing policy 9")
    # S, t = test_policy_short(env, q_table)
    # print("Testing policy 10")
    # S, t = test_policy_short(env, q_table)



if __name__ == '__main__':
    main()
