# TODO copy q-policy coz of pycharm bug

import itertools
import sys
from collections import defaultdict

import numpy as np

from environments.arm_env.arm_env import ArmEnv
from environments.env_utils import EnvForTesting
from environments.grid_maze_env.grid_maze_generator import generate_maze_please
from environments.grid_maze_env.maze_world_env import MazeWorldEpisodeLength
from utils.plotting import plot_multi_test


def arg_max_action(q_dict, state, action_space):
    result_action = 0
    for action_to in range(action_space):
        if q_dict[state, action_to] > q_dict[state, result_action]:
            result_action = action_to
    return result_action


def q_learning(env, num_episodes, eps=0.1, alpha=0.1, gamma=0.9, q_table=None):
    to_plot = []

    # initialize q-function
    if q_table is None:
        q_table = defaultdict(lambda: 0)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        ep_reward = 0
        if (i_episode + 1) % 10 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
            eps = eps - 0.01 * eps

        # Reset the environment and pick the first state
        state = env.reset()

        for t in itertools.count():
            # E-greedy
            if np.random.rand(1) < eps:
                # choosing a random action
                action = np.random.choice(env.action_space.n, size=1)[0]
            else:
                # choosing arg_max action
                action = arg_max_action(q_dict=q_table, state=state, action_space=env.action_space.n)

            next_state, reward, done, _ = env.step(action)
            # print(q_table[state, action])
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (
                reward + gamma * q_table[next_state, arg_max_action(q_dict=q_table, state=next_state, action_space=env.action_space.n)])

            # Update statistics
            ep_reward += reward

            if done:
                break

            state = next_state
        to_plot.append(ep_reward)
    return to_plot, q_table


def check_policy(env, q_table):
    state = env.reset()
    s_r = 0
    s_t = 0

    for t in itertools.count():
        # WE CAN PRINT ENVIRONMENT STATE
        env.render()

        # Take a step
        action = arg_max_action(q_dict=q_table, state=state, action_space=env.action_space.n)
        # action = np.argmax(q_table[state, :])
        next_state, reward, done, _ = env.step(action)

        # Update statistics
        s_r += reward
        s_t = t

        if done:
            break

        state = next_state
    return s_r, s_t


def random_policy(env, num_episodes, ):
    to_plot = []

    for i_episode in range(num_episodes):
        ep_reward = 0
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        env.reset()
        for t in itertools.count():
            action = np.random.choice(env.action_space.n, size=1)[0]
            next_state, reward, done, _ = env.step(action)

            ep_reward += reward

            if done:
                break
        to_plot.append(ep_reward)
    return to_plot


def grid_maze_env():
    env = MazeWorldEpisodeLength(maze=generate_maze_please())
    ep_length = 2000

    # q-learning policy on MazeWorldEpisodeLength
    q_stats, q_table = q_learning(env, ep_length)
    s, t = check_policy(env, q_table)
    print(s, t)

    # random policy on MazeWorldEpisodeLength
    r_stats = random_policy(env, ep_length)

    plot_multi_test([q_stats, r_stats])


def arm_env():
    env = ArmEnv(episode_max_length=300,
                 size_x=5,
                 size_y=3,
                 cubes_cnt=4,
                 action_minus_reward=-1,
                 finish_reward=100,
                 tower_target_size=4)

    ep_length = 800
    # q-learning policy on MazeWorldEpisodeLength
    q_stats, q_table = q_learning(env, ep_length)
    s, t = check_policy(env, q_table)
    print(s, t)

    # random policy on MazeWorldEpisodeLength
    r_stats = random_policy(env, ep_length)

    plot_multi_test([q_stats, r_stats])


def main():
    grid_maze_env()
    arm_env()


if __name__ == '__main__':
    global_env = EnvForTesting()
    q_stats, q_table = q_learning(env=global_env.env, num_episodes=global_env.episodes)
    print("\ntotal_reward", sum(q_stats))
    # plot_multi_test([q_stats])
