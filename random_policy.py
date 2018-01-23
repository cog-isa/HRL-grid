import itertools
import sys

import numpy as np

from environments.arm_env.arm_env import ArmEnv
from environments.grid_maze_env.grid_maze_generator import generate_maze_please
from environments.grid_maze_env.maze_world_env import MazeWorldEpisodeLength
from utils import plotting


def random_policy(env, num_episodes, ):
    statistics = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        for t in itertools.count():
            action = np.random.choice(env.action_space.n, size=1)[0]
            next_state, reward, done, _ = env.step(action)

            statistics.episode_rewards[i_episode] += reward
            statistics.episode_lengths[i_episode] = t

            if done:
                break

    return statistics


def main():
    env = MazeWorldEpisodeLength(maze=generate_maze_please())
    env = ArmEnv(size_x=12,)
    stats = random_policy(env, 100)
    plotting.plot_episode_stats(stats)


if __name__ == '__main__':
    main()
