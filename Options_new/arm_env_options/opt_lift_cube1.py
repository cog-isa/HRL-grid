import itertools
import pickle
import sys

import numpy as np
from Options_new.arm_env_options.baseline_q_table import q_learning
from Options_new.arm_env_options.opt_lift_cube_init import test_policy_opt
from environments.arm_env.arm_env import ArmEnv
from utils import plotting


def q_learning_on_options(env, option_q_table, init_states, term_states, num_episodes, eps=0.1, alpha=0.4, gamma=1.0):

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    n_actions = env.action_space.n

    # initialize q-function
    q_table = {}

    for i_episode in range(num_episodes):

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

            # Take an action
            if np.random.rand(1) < eps:  # choose random option
                # action0 = np.random.choice(np.arange(n_actions + 1*(state in init_states)), size=1)[0]
                action0 = np.random.choice([0, 2, 4, 5])
            else:
                # action0 = np.argmax(q_table[state])
                action0 = 0
                for l in range(2, len(q_table[state])):
                    if l != 3 and q_table[state][l] > q_table[state][action0]:
                        action0 = l

            # if i_episode == num_episodes - 1: print(action0)
            # if action0 == 1 or action0 == 3:
            #     print("\n smth wrong")
            #     print(q_table[state][3])

            # if option is chosen
            if action0 >= n_actions:
                # print("\n Option is chosen \n")
                # flag = 1
                # env.render()
                opt_rew = 0
                opt_t = 0
                opt_state = state

                while True:

                    action = np.argmax(option_q_table[opt_state])
                    opt_state, reward, done, _ = env._step(action, options=True)

                    opt_rew += reward
                    opt_t += 1

                    if done or opt_state in term_states or opt_t >= 20:
                        break

                env._episode_length += 1

                # opt_rew += 5

                next_state = opt_state
                if next_state not in q_table:
                    q_table[next_state] = np.zeros(shape=n_actions+1*(next_state in init_states))

                z = q_table[next_state][0]
                for l in range(2, len(q_table[next_state])):
                    if l != 3 and q_table[next_state][l] > z:
                        z = q_table[next_state][l]

                q_table[state][action0] = (1 - alpha) * q_table[state][action0] + alpha * (
                    opt_rew + gamma ** opt_t * z)

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

                z = q_table[next_state][0]
                for l in range(2, len(q_table[next_state])):
                    if l != 3 and q_table[next_state][l] > z:
                        z = q_table[next_state][l]

                q_table[state][action0] = (1 - alpha) * q_table[state][action0] + alpha * (
                    reward + gamma * z)

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


def main():
    env = ArmEnv(episode_max_length=200,
                 size_x=8,
                 size_y=4,
                 cubes_cnt=6,
                 action_minus_reward=-1,
                 finish_reward=200,
                 tower_target_size=6)

    with open('opt_q_table.txt', 'rb') as handle:
        opt_q_table = pickle.loads(handle.read())

    with open('opt_init_st.txt', 'rb') as handle:
        init_states = pickle.loads(handle.read())

    with open('opt_term_st.txt', 'rb') as handle:
        term_states = pickle.loads(handle.read())

    # print(len(opt_q_table))
    # print(opt_q_table)

    stats2, q_table2 = q_learning_on_options(env, opt_q_table, init_states, term_states, 8000)

    plotting.plot_episode_stats(stats2)
    #
    # print(q_table2)
    # print(len(q_table2))

    S, t = test_policy_opt(env, q_table2, opt_q_table, init_states, term_states)

    # stats3, q_table3 = q_learning(env, 8000)
    #
    # plotting.plot_multi_test(smoothing_window=30,
    #                          x_label="episode",
    #                          y_label="smoothed rewards",
    #                          curve_to_draw=[stats2.episode_rewards,
    #                                         stats3.episode_rewards],
    #                          labels=["options", "q-learning"]
    #                          )

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