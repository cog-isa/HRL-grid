import sys
from collections import defaultdict

import matplotlib
import numpy as np

from HAM.basic_machine import root_machine
from environments.grid_maze_generator import generate_maze_please
from environments.maze_world_env import MazeWorld
from lib import plotting
from HAM.utils import choice_update, apply_action, who_a_mi


def ham_learning(env, num_episodes, discount_factor=0.8, alpha=0.1, epsilon=0.1):
    Q = defaultdict(lambda: defaultdict(lambda: 0))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following

    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        info = {"Q": Q,
                "env": env,
                "r": 0,
                "gamma": 0.8,
                "last": None,
                "total_reward": 0,
                "actions_cnt": 0,
                "done": False,
                "state": state,
                "EPSILON": 0.1,
                "DIS_FACTOR": 0.8,
                "ALPHA": 0.1,
                "stats": stats,
                "choice_update": choice_update,
                "apply_action": apply_action,
                "me": who_a_mi,
                }
        info = root_machine.start(info)
        stats.episode_lengths[i_episode] = info["actions_cnt"]
        stats.episode_rewards[i_episode] += info["total_reward"]

    return Q, stats


matplotlib.style.use('ggplot')

env = MazeWorld(maze=generate_maze_please())
Q, stats = ham_learning(env, 500)
plotting.plot_episode_stats(stats)
