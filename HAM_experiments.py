from environments.maze_world_env import MazeWorld
from lib import plotting
from HAM.utils import ham_learning
from HAM.ham_machine import *
from environments.grid_maze_generator import *

base_patterns = [2 ** 4 + 2 ** 8, 1 + 2 ** 12, 0]
x = list(map(generate_pattern, base_patterns))


def test1():
    mz_level1 = generate_maze(x, size_x=3, size_y=3, seed=15)
    mz_level2 = generate_maze([mz_level1], size_x=3, size_y=3)
    return place_start_finish(prepare_maze(mz_level2))


def test2():
    mz_level1 = generate_maze(deepcopy(x), size_x=4, size_y=4, seed=95)
    mz_level2 = generate_maze([mz_level1], size_x=4, size_y=4, seed=1)
    return place_start_finish(prepare_maze(mz_level2))


def test3():
    mz_level1 = generate_maze(x, size_x=3, size_y=3, seed=65)
    mz_level2 = generate_maze([mz_level1], size_x=3, size_y=3, seed=1)
    place_start_finish(prepare_maze(mz_level2))


def test4():
    mz_level1 = generate_maze(x, size_x=3, size_y=3, seed=95)
    mz_level2 = generate_maze([mz_level1], size_x=3, size_y=3, seed=1)
    return place_start_finish(prepare_maze(mz_level2))


def experiment_01():
    # in this experiment, the agent is looping on the 500th iteration
    # but with discount_factor = 0.9 - all ok
    env = MazeWorld(maze=test1())
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
    env = MazeWorld(maze=test1())
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
    env = MazeWorld(maze=test1())
    env.render()
    params = {
        "env": env,
        "num_episodes": 100,
        "machine": l2_machine,
        "alpha": 0.1,
        "epsilon": 0.1,
        "discount_factor": 0.9,
    }
    Q, stats = ham_learning(**params)

    env = MazeWorld(maze=test1())
    env.render()
    params = {
        "env": env,
        "num_episodes": 100,
        "machine": BasicMachine,
        "alpha": 0.2,
        "epsilon": 0.2,
        "discount_factor": 1,
    }
    Q1, stats1 = ham_learning(**params)
    plotting.plot_multi_test(curve_to_draw=[stats.episode_rewards, stats1.episode_rewards])


def experiment_04():
    env = MazeWorld(maze=test1(), finish_reward=1000)
    env.render()
    params = {
        "env": env,
        "num_episodes": 200,
        "machine": l1_machine,
        "alpha": 0.1,
        "epsilon": 0.1,
        "discount_factor": 0.9,
    }
    Q, stats = ham_learning(**params)

    env = MazeWorld(maze=test1(), finish_reward=1000)
    env.render()
    params = {
        "env": env,
        "num_episodes": 200,
        "machine": BasicMachine,
        "alpha": 0.1,
        "epsilon": 0.5,
        "discount_factor": 0.9,
    }
    Q1, stats1 = ham_learning(**params)

    env = MazeWorld(maze=test1(), finish_reward=1000)
    env.render()
    params = {
        "env": env,
        "num_episodes": 200,
        "machine": BasicMachine,
        "alpha": 0.1,
        "epsilon": 0.2,
        "discount_factor": 0.9,
    }
    Q2, stats2 = ham_learning(**params)

    plotting.plot_multi_test(curve_to_draw=[stats.episode_rewards, stats1.episode_rewards, stats2.episode_rewards])


if __name__ == "__main__":
    experiment_04()
