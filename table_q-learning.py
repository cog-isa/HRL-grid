import itertools
import sys

import numpy as np

from environments.arm_env.arm_env import ArmEnv
from environments.grid_maze_env.grid_maze_generator import generate_maze_please
from environments.grid_maze_env.maze_world_env import MazeWorld
from utils import plotting


def q_learning(env, num_episodes, eps=0.4, alpha=0.1, gamma=0.8):
    to_plot = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # initialize q-function
    q_table = np.zeros(shape=(env.observation_space.n, env.action_space.n))

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.

        if (i_episode + 1) % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
            eps = eps - 0.01 * eps

        # Reset the environment and pick the first action
        state = env.reset()

        for t in itertools.count():
            # WE CAN PRINT ENVIRONMENT STATE
            # env.render()

            # Take a step
            if np.random.rand(1) < eps:  # choose random action
                action = np.random.choice(env.action_space.n, size=1)[0]
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, done, _ = env.step(action)
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state, :]))

            # Update statistics
            to_plot.episode_rewards[i_episode] += reward
            to_plot.episode_lengths[i_episode] = t

            if done:
                break

            state = next_state

    return to_plot, q_table


def test_policy(env, q_table):
    state = env.reset()
    S_r = 0
    S_t = 0

    for t in itertools.count():
        # WE CAN PRINT ENVIRONMENT STATE
        env.render()

        # Take a step
        action = np.argmax(q_table[state, :])
        next_state, reward, done, _ = env.step(action)

        # Update statistics
        S_r += reward
        S_t = t

        if done:
            break

        state = next_state
    return S_r, S_t


def main():
    env = ArmEnv(episode_max_length=300,
                 size_x=5,
                 size_y=3,
                 cubes_cnt=4,
                 action_minus_reward=-1,
                 finish_reward=100,
                 tower_target_size=4)
    env = MazeWorld(maze=generate_maze_please())

    # make 50 iterations
    stats, q_table = q_learning(env, 20000)
    plotting.plot_episode_stats(stats)

    s, t = test_policy(env, q_table)
    print(s, t)

if __name__ == '__main__':
    main()
