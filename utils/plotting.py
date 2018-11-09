import pandas as pd
from collections import namedtuple
# from matplotlib import pyplot as plt
import matplotlib

# matplotlib.use('agg')
import matplotlib.pyplot as plt

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def plot_episode_stats(stats, smoothing_window=10, no_show=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if no_show:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if no_show:
        plt.close(fig2)
    else:
        plt.show(fig2)


def plot_multi_test(curve_to_draw=None, smoothing_window=10, x_label="X", y_label="Y", labels=None, filename=None):
    fig2 = plt.figure(figsize=(10, 5))

    t = []
    for index, elem in enumerate(curve_to_draw):
        rewards_smoothed = pd.Series(elem).rolling(smoothing_window, min_periods=smoothing_window).mean()
        p, = plt.plot(rewards_smoothed)
        t.append(p)
    plt.legend(t, labels) if labels else plt.legend(t)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # plt.savefig("diagram.png" if filename is None else filename + ".png")
    plt.show()
