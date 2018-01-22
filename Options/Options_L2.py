import itertools
import random
import sys

import matplotlib
import numpy as np

from environments.grid_maze_env.grid_maze_generator import generate_maze_please
from environments.grid_maze_env.maze_world_env import MazeWorldEpisodeLength
from lib import plotting

matplotlib.style.use('ggplot')

class option():
    def __init__(self, env, init_set, term_cond, goal_state, q_table,d):
        self.env = env
        self.init_set = init_set
        self.term_cond = term_cond
        self.goal_state = goal_state
        self.q_table = q_table
        self.d = d

    def learning_option(self, num_episodes=50, eps=0.4, alpha=0.1, gamma=0.8):

        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

        for i_episode in range(num_episodes):

            if (i_episode + 1) % 50 == 0:
                eps = eps - 0.01 * eps

            # Random choice of the initial position
            state = self.env.reset()
            self.env.s = random.choice(self.init_set)
            state = self.env.s

            for t in itertools.count():

                # Take a step
                if np.random.rand(1) < eps:  # choose random action
                    action = np.random.choice(self.env.action_space.n, size=1)[0]
                else:
                    action = np.argmax(self.q_table[self.d[state], :])

                next_state, reward, done, _ = self.env.step(action)

                if self.term_cond[next_state] == 1.0:
                    done = True

                if self.goal_state == next_state:
                    reward += 50

                self.q_table[self.d[state], action] = (1 - alpha) * self.q_table[self.d[state], action] + alpha * (
                    reward + gamma * np.max(self.q_table[self.d[next_state], :]))

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                if done:
                    break

                state = next_state

        return stats, self.q_table

env = MazeWorldEpisodeLength(maze=generate_maze_please(size_x=2, size_y=2))

#  опция выход через верхний проход для правого нижнего паттерна
init_set = np.array([22,26,27,28,32,33,34,38,39,40])
term_cond = np.ones(shape = (env.observation_space.n,))
term_cond[init_set] = 0
goal_state = 20
q_table = np.zeros(shape=(init_set.shape[0]+2, env.action_space.n))
d = dict(zip(np.append(init_set, [31,20]), np.arange(init_set.shape[0]+2))) # to map q-table rows to the right states

option1 = option(env, init_set= init_set, term_cond=term_cond, goal_state=goal_state, q_table=q_table,d=d)
_, _ = option1.learning_option(num_episodes=70)

#  опция выход через левый проход для правого нижнего паттерна
init_set2 = np.array([22,26,27,28,32,33,34,38,39,40])
term_cond2 = np.ones(shape = (env.observation_space.n,))
term_cond2[init_set2] = 0
goal_state2 = 31
q_table2 = np.zeros(shape=(init_set2.shape[0]+2, env.action_space.n))
d2 = dict(zip(np.append(init_set2, [20,31]), np.arange(init_set2.shape[0]+2))) # to map q-table rows to the right states

option2 = option(env, init_set=init_set2, term_cond=term_cond2, goal_state=goal_state2, q_table=q_table2, d=d2)
_, _ = option2.learning_option(num_episodes=100)

#  опция выход через верхний проход для левого нижнего паттерна
init_set3 = np.array([21,23,24,25,29,30,31,35,36,37])
term_cond3 = np.ones(shape = (env.observation_space.n,))
term_cond3[init_set3] = 0
goal_state3 = 19
q_table3 = np.zeros(shape=(init_set3.shape[0]+2, env.action_space.n))
d3 = dict(zip(np.append(init_set3, [32,19]), np.arange(init_set3.shape[0]+2))) # to map q-table rows to the right states

option3 = option(env, init_set=init_set3, term_cond=term_cond3, goal_state=goal_state3, q_table=q_table3, d=d3)
_, _ = option3.learning_option(num_episodes=100)

#  опция выход через правый проход для левого нижнего паттерна
init_set4 = np.array([21,23,24,25,29,30,31,35,36,37])
term_cond4 = np.ones(shape = (env.observation_space.n,))
term_cond4[init_set4] = 0
goal_state4 = 32
q_table4 = np.zeros(shape=(init_set4.shape[0]+2, env.action_space.n))
d4 = dict(zip(np.append(init_set4, [19,32]), np.arange(init_set4.shape[0]+2))) # to map q-table rows to the right states

option4 = option(env, init_set=init_set4, term_cond=term_cond4, goal_state=goal_state4, q_table=q_table4, d=d4)
_, _ = option4.learning_option(num_episodes=100)

#  опция выход через левый проход для правого верхнего паттерна
init_set5 = np.array([4,5,6,10,11,12,16,17,18,20])
term_cond5 = np.ones(shape = (env.observation_space.n,))
term_cond5[init_set5] = 0
goal_state5 = 9
q_table5 = np.zeros(shape=(init_set5.shape[0]+2, env.action_space.n))
d5 = dict(zip(np.append(init_set5, [9,22]), np.arange(init_set5.shape[0]+2))) # to map q-table rows to the right states

option5 = option(env, init_set=init_set5, term_cond=term_cond5, goal_state=goal_state5, q_table=q_table5, d=d5)
_, _ = option5.learning_option(num_episodes=100)

#  опция выход через нижний проход для правого верхнего паттерна
init_set6 = np.array([4,5,6,10,11,12,16,17,18,20])
term_cond6 = np.ones(shape=(env.observation_space.n,))
term_cond6[init_set6] = 0
goal_state6 = 22
q_table6 = np.zeros(shape=(init_set6.shape[0]+2, env.action_space.n))
d6 = dict(zip(np.append(init_set6, [9,22]), np.arange(init_set6.shape[0]+2))) # to map q-table rows to the right states

option6 = option(env, init_set=init_set6, term_cond=term_cond6, goal_state=goal_state6, q_table=q_table6, d=d6)
_, _ = option6.learning_option(num_episodes=100)



def q_learning_on_options(env, options, num_episodes, eps=0.4, alpha=0.1, gamma=0.8):
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    n_actions = env.action_space.n
    n_states = env.observation_space.n
    q = []
    for i in range(1,n_states):
        if i in [1,2,3,7,8,9,13,14,15,19]:
            b = [e for e in range(n_actions)] #basic moves are available everywhere
        else: b = []
        for k, o_ in enumerate(options):
            if i in o_.init_set:
                b.append(n_actions+k)
        q.append(b)



    # initialize q-function
    q_table = np.zeros(shape=(env.observation_space.n, env.action_space.n + len(options)))

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.

        if (i_episode + 1) % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
            eps = eps - 0.01 * eps

        # Reset the environment
        state = env.reset()

        for t in itertools.count():
            # WE CAN PRINT ENVIRONMENT STATE
            if i_episode == num_episodes - 1:
                print(state)
                env.render()
                print(action0)

            # Take an action
            if np.random.rand(1) < eps:  # choose random option
                action0 = np.random.choice(q[state-1])
            else:
                #action0 = np.where(q_table[state, :] == np.max(q_table[state, q[state-1]]))[0][0]
                argmax = q[state-1][0]
                for l in q[state-1][1:]:
                    if q_table[state, l] > q_table[state, argmax]:
                        argmax = l
                action0 = argmax


            if action0 >= 4:
                opt = options[action0 - n_actions]
                #execute the option's policy
                opt_rev = 0
                opt_t = 0
                opt_state = state

                while True:

                    action = np.argmax(opt.q_table[opt.d[opt_state], :])
                    opt_state, reward, done, _ = env.step(action)

                    opt_rev += reward
                    opt_t += 1

                    if done or opt_state == opt.goal_state:
                        break

                next_state = opt_state
                reward = opt_rev

                #update rule in case of options
                q_table[state, action0] = (1 - alpha) * q_table[state, action0] + alpha * (
                opt_rev + gamma**opt_t * np.max(q_table[next_state, q[next_state-1]]))

                # Update statistics
                stats.episode_rewards[i_episode] += opt_rev
                stats.episode_lengths[i_episode] = t

                if done:
                    break

                state = next_state

            else:
                next_state, reward, done, _ = env.step(action0)
                q_table[state, action0] = (1 - alpha) * q_table[state, action0] + alpha * (
                    reward + gamma * np.max(q_table[next_state, :]))

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                if done:
                    break

                state = next_state

    return stats, q_table

stats, q_table = q_learning_on_options(env, [option1,option2,option3, option4, option5,option6],200)
plotting.plot_episode_stats(stats)
print(q_table)

def test_policy(env, q_table, options):
    state = env.reset()
    s_r = 0
    s_t = 0
    n_actions = env.action_space.n

    for t in itertools.count():
        # WE CAN PRINT ENVIRONMENT STATE
        env.render()
        print(state)

        # Take a step
        action = np.argmax(q_table[state, :])

        if action >= 4:
            opt = options[action - n_actions]
            # execute the option's policy
            opt_rev = 0
            opt_t = 0
            opt_state = state

            while True:

                action = np.argmax(opt.q_table[opt.d[opt_state], :])
                opt_state, reward, done, _ = env.step(action)

                opt_rev += reward
                opt_t += 1

                if done or opt_state == opt.goal_state:
                    break

            next_state = opt_state
            reward = opt_rev

            s_r += reward
            s_t = t

            if done:
                break

            state = next_state

        else:
            next_state, reward, done, _ = env.step(action)

            s_r += reward
            s_t = t

            if done:
                break

        state = next_state
    return s_r, s_t

s, t = test_policy(env, q_table, [option1, option2, option3, option4, option5, option6])
print(s, t)

