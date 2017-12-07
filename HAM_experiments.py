from collections import defaultdict
import sys
from environments.maze_world_env import MazeWorld, MazeWorldEpisodeLength
from lib import plotting
from HAM.utils import ham_learning
from HAM.machines import *
from environments.grid_maze_generator import *
import threading

base_patterns = [2 ** 4 + 2 ** 8, 1 + 2 ** 12, 0]
x = list(map(generate_pattern, base_patterns))


def input_01():
    mz_level1 = generate_maze(x, size_x=3, size_y=3, seed=15)
    mz_level2 = generate_maze([mz_level1], size_x=3, size_y=3)
    return place_start_finish(prepare_maze(mz_level2))


def input_02():
    mz_level1 = generate_maze(deepcopy(x), size_x=4, size_y=4, seed=95)
    mz_level2 = generate_maze([mz_level1], size_x=4, size_y=4, seed=1)
    return place_start_finish(prepare_maze(mz_level2))


def input_03():
    mz_level1 = generate_maze(x, size_x=3, size_y=3, seed=65)
    mz_level2 = generate_maze([mz_level1], size_x=3, size_y=3, seed=1)
    place_start_finish(prepare_maze(mz_level2))


def input_04():
    mz_level1 = generate_maze(x, size_x=3, size_y=3, seed=95)
    mz_level2 = generate_maze([mz_level1], size_x=3, size_y=3, seed=1)
    return place_start_finish(prepare_maze(mz_level2))


def input_05():
    mz_level1 = generate_maze(x, size_x=3, size_y=3, seed=15)
    mz_level2 = generate_maze([mz_level1], size_x=2, size_y=2)
    return place_start_finish(prepare_maze(mz_level2))


def experiment_01():
    # in this experiment, the agent is looping on the 500th iteration
    # but with discount_factor = 0.9 - all ok
    env = MazeWorld(maze=input_01())
    env.render()
    params = {
        "env": env,
        "num_episodes": 2000,
        "machine": BasicMachine,
        "alpha": 0.1,
        "epsilon": 0.1,
        "discount_factor": 0.8,
    }
    Q, stats = ham_learning(**params)
    plotting.plot_episode_stats(stats)


def experiment_02():
    # in this experiment, the agent is looping on the 500th iteration
    # but with discount_factor = 0.9 - all ok
    env = MazeWorld(maze=input_01())
    env.render()
    params = {
        "env": env,
        "num_episodes": 2000,
        "machine": BasicMachine,
        "alpha": 0.1,
        "epsilon": 0.1,
        "discount_factor": 0.9,
    }
    Q, stats = ham_learning(**params)
    plotting.plot_episode_stats(stats)


def experiment_03():
    env = MazeWorld(maze=input_01(), finish_reward=1000000)
    env.render()
    params = {
        "env": env,
        "num_episodes": 300,
        "machine": L2Move5,
        "alpha": 0.1,
        "epsilon": 0.1,
        "discount_factor": 0.9,
    }
    Q, stats = ham_learning(**params)

    env = MazeWorld(maze=input_01(), finish_reward=1000000)
    env.render()
    params = {
        "env": env,
        "num_episodes": 400,
        "machine": BasicMachine,
        "alpha": 0.1,
        "epsilon": 0.1,
        "discount_factor": 0.9,
    }
    Q1, stats1 = ham_learning(**params)

    plotting.plot_multi_test(curve_to_draw=[stats.episode_rewards, stats1.episode_rewards])


def experiment_04():
    env = MazeWorld(maze=input_01(), finish_reward=1000)
    env.render()
    params = {
        "env": env,
        "num_episodes": 250,
        "machine": L1Machine,
        "alpha": 0.1,
        "epsilon": 0.1,
        "discount_factor": 0.9,
    }
    Q, stats = ham_learning(**params)

    env = MazeWorld(maze=input_01(), finish_reward=1000)
    env.render()
    params = {
        "env": env,
        "num_episodes": 300,
        "machine": BasicMachine,
        "alpha": 0.1,
        "epsilon": 0.5,
        "discount_factor": 0.9,
    }
    Q1, stats1 = ham_learning(**params)

    env = MazeWorld(maze=input_01(), finish_reward=1000)
    env.render()
    params = {
        "env": env,
        "num_episodes": 450,
        "machine": BasicMachine,
        "alpha": 0.1,
        "epsilon": 0.2,
        "discount_factor": 0.9,
    }
    Q2, stats2 = ham_learning(**params)

    plotting.plot_multi_test(curve_to_draw=[stats.episode_rewards, stats1.episode_rewards, stats2.episode_rewards])


# def experiment_05():
#     env1 = MazeWorld(maze=input_02(), finish_reward=1000)
#     env1.render()
#     params1 = {
#         "env": env1,
#         "num_episodes": 300,
#         "machine": L2Move5,
#         "alpha": 0.1,
#         "epsilon": 0.1,
#         "discount_factor": 1,
#     }
#     t1 = threading.Thread(target=ham_learning, kwargs=params1)
#
#     env2 = MazeWorld(maze=input_02(), finish_reward=1000)
#     env2.render()
#     params2 = {
#         "env": env2,
#         "num_episodes": 400,
#         "machine": BasicMachine,
#         "alpha": 0.2,
#         "epsilon": 0.2,
#         "discount_factor": 1,
#     }
#     t2 = threading.Thread(target=ham_learning, kwargs=params2)
#
#   q2, stats2 = t2.start()
#
#   t1.join()
#   t2.join()
#   plotting.plot_multi_test(curve_to_draw=[stats1.episode_rewards, stats2.episode_rewards])


def experiment_06():
    env = MazeWorld(maze=input_01(), finish_reward=100000)
    env.render()
    params = {
        "env": env,
        "num_episodes": 600,
        "machine": L2Move3,
        "alpha": 0.1,
        "epsilon": 0.1,
        "discount_factor": 0.9,
    }
    Q, stats = ham_learning(**params)

    env = MazeWorld(maze=input_01(), finish_reward=100000)
    env.render()
    params = {
        "env": env,
        "num_episodes": 610,
        "machine": L2Move5,
        "alpha": 0.1,
        "epsilon": 0.1,
        "discount_factor": 0.9,
    }
    Q1, stats1 = ham_learning(**params)

    env = MazeWorld(maze=input_01(), finish_reward=100000)
    env.render()
    params = {
        "env": env,
        "num_episodes": 620,
        "machine": BasicMachine,
        "alpha": 0.1,
        "epsilon": 0.1,
        "discount_factor": 0.9,
    }
    Q2, stats2 = ham_learning(**params)

    plotting.plot_multi_test(curve_to_draw=[stats.episode_rewards, stats1.episode_rewards, stats2.episode_rewards])


def experiment_07():
    finish_reward = 10000
    episode_max_length = 1000
    maze = input_01()
    episodes_count = 500

    env = MazeWorldEpisodeLength(maze=deepcopy(maze), finish_reward=finish_reward,
                                 episode_max_length=episode_max_length)
    m1 = {"env": env, "num_episodes": episodes_count, "machine": L2Move3, }
    m2 = {"env": env, "num_episodes": episodes_count, "machine": L2Move5, }
    m3 = {"env": env, "num_episodes": episodes_count, "machine": BasicMachine, }
    Q1, stats1 = ham_learning(**m1)
    Q2, stats2 = ham_learning(**m2)
    Q3, stats3 = ham_learning(**m3)

    plotting.plot_multi_test(smoothing_window=30,
                             xlabel="episode",
                             ylabel="smoothed rewards",
                             curve_to_draw=[stats1.episode_rewards, stats2.episode_rewards, stats3.episode_rewards],
                             labels=[m1["machine"], m2["machine"], m3["machine"]]
                             )
    st1 = []
    episode_sum = 0
    for i in stats1.episode_rewards:
        episode_sum += i * 2 >= finish_reward
        st1.append(episode_sum)
    st2 = []
    episode_sum = 0
    for i in stats2.episode_rewards:
        episode_sum += i * 2 >= finish_reward
        st2.append(episode_sum)

    st3 = []
    episode_sum = 0
    for i in stats3.episode_rewards:
        episode_sum += i * 2 >= finish_reward
        st3.append(episode_sum)

    plotting.plot_multi_test(smoothing_window=10,
                             xlabel="episode",
                             ylabel="done cnt",
                             curve_to_draw=[st1, st2, st3],
                             labels=[m1["machine"], m2["machine"], m3["machine"]]
                             )


def experiment_08():
    finish_reward = 10000
    episode_max_length = 500
    maze = input_05()
    q1_result = []
    q2_result = []
    q3_result = []
    Q1 = defaultdict(lambda: defaultdict(lambda: 0))
    Q2 = defaultdict(lambda: defaultdict(lambda: 0))
    Q3 = defaultdict(lambda: defaultdict(lambda: 0))

    env = MazeWorldEpisodeLength(maze=deepcopy(maze), finish_reward=finish_reward,
                                 episode_max_length=episode_max_length)
    ep_size = 5000
    for i in range(ep_size):
        print("\r episode {i}/{ep_size}.".format(**locals()), end="")
        sys.stdout.flush()
        episodes_count = 1

        m1 = {"env": env, "num_episodes": episodes_count, "machine": L2Move3, "q": Q1}
        m2 = {"env": env, "num_episodes": episodes_count, "machine": L2Move5, "q": Q2}
        m3 = {"env": env, "num_episodes": episodes_count, "machine": BasicMachine, "q": Q3}
        _, stats1 = ham_learning(**m1)
        _, stats2 = ham_learning(**m2)
        _, stats3 = ham_learning(**m3)
        m1 = {"env": env, "num_episodes": 1, "machine": L2Move3, "q": Q1, "alpha": 0, "epsilon": 0}
        m2 = {"env": env, "num_episodes": 1, "machine": L2Move5, "q": Q2, "alpha": 0, "epsilon": 0}
        m3 = {"env": env, "num_episodes": episodes_count, "machine": BasicMachine, "q": Q3, "epsilon": 0,
              "alpha": 0}
        _, stats1 = ham_learning(**m1)
        _, stats2 = ham_learning(**m2)
        _, stats3 = ham_learning(**m3)
        q1_result.append(stats1.episode_rewards[0])
        q2_result.append(stats2.episode_rewards[0])
        q3_result.append(stats3.episode_rewards[0])

    m1 = {"machine": L2Move3}
    m2 = {"machine": L2Move5}
    m3 = {"machine": BasicMachine}

    plotting.plot_multi_test(smoothing_window=100,
                             xlabel="episode",
                             ylabel="smoothed rewards",
                             curve_to_draw=[q1_result, q2_result, q3_result],
                             labels=[m1["machine"], m2["machine"], m3["machine"]]
                             )


if __name__ == "__main__":
    experiment_08()
