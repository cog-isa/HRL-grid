from HAM_new.HAM_core import AutoBasicMachine, RootMachine, LoopInvokerMachine, AbstractMachine, Start, Choice, Action, Stop, MachineRelation, Call
from HAM_new.HAM_experiments.HAM_utils import HAMParamsCommon, maze_world_input_01, plot_multi, ham_runner, PlotParams
from environments.grid_maze_env.maze_world_env import MazeWorldEpisodeLength

to_plot = []
env = MazeWorldEpisodeLength(maze=maze_world_input_01(), episode_max_length=300)
num_episodes = 5300

params = HAMParamsCommon(env)
ham_runner(ham=AutoBasicMachine(env), num_episodes=num_episodes, env=env, params=params)
to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="HAM_basic"))

# --------------------------------------------------------
# TODO code l2interesting machine from old_ham

call = Call(machine_to_call=AutoBasicMachine(env=env))
transitions = (
    MachineRelation(left=Start(), right=call),
    MachineRelation(left=call, right=Stop()),

)

ololo_machine = AbstractMachine(transitions=transitions)

params = HAMParamsCommon(env)
ham_runner(ham=ololo_machine, num_episodes=num_episodes, env=env, params=params)
to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="HAM_with_pull_up"))

plot_multi(to_plot)
