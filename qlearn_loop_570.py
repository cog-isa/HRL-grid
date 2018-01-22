import itertools
import sys

import matplotlib
import numpy as np

from environments.grid_maze_env.grid_maze_generator import generate_maze, generate_pattern, prepare_maze, place_start_finish
from environments.grid_maze_env.maze_world_env import MazeWorld
from lib import plotting


def q_learning(env, num_episodes, eps=0.01, alpha=0.1, gamma=0.5):
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # initialize q-function
    q_table = np.zeros(shape=(env.observation_space.n, env.action_space.n))

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.

        if (i_episode + 1) % 10 == 0:
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
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state, :]))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break

            state = next_state

    return stats, q_table


base_patterns = [2 ** 4 + 2 ** 8, 1 + 2 ** 12, 0]
x = list(map(generate_pattern, base_patterns))


def input_01():
    mz_level1 = generate_maze(x, size_x=3, size_y=3, seed=15)
    mz_level2 = generate_maze([mz_level1], size_x=3, size_y=3)
    return place_start_finish(prepare_maze(mz_level2))


matplotlib.style.use('ggplot')

env = MazeWorld(maze=input_01())
# make 50 iterations
stats, q_table = q_learning(env, 20000)
plotting.plot_episode_stats(stats)

