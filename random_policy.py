import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

from collections import defaultdict
from grid_maze_env import MazeWorld
import plotting

from gride_generator import generate_maze, generate_pattern, prepare_maze, generate_maze_please


def random_policy(env, num_episodes, ):
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.

        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        for t in itertools.count():
            # WE CAN PRINT ENVIRONMENT STATE
            env.render()

            # Take a step
            action = np.random.choice(env.action_space.n, size=1)[0]
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break

            state = next_state

    return stats




matplotlib.style.use('ggplot')

env = MazeWorld(maze=generate_maze_please())
# make 50 iterations
stats = random_policy(env, 50)
env._render()
plotting.plot_episode_stats(stats)
