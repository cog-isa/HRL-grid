from HAM_new.HAM_core import AutoBasicMachine
from HAM_new.HAM_experiments.HAM_utils import HAMParamsCommon, maze_world_input_01, plot_multi, ham_runner, PlotParams
from environments.grid_maze_env.maze_world_env import MazeWorld

to_plot = []
env = MazeWorld(maze_world_input_01())

params = HAMParamsCommon(env)
ham_runner(ham=AutoBasicMachine(env), num_episodes=300, env=env, params=params)
to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="HAM_basic"))

params = HAMParamsCommon(env)
ham_runner(ham=AutoBasicMachine(env), num_episodes=300, env=env, params=params)
to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="HAM_basic"))

plot_multi(to_plot)
