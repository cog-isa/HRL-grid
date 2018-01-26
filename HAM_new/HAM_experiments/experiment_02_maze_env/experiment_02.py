from HAM_new.HAM_core import AutoBasicMachine, AbstractMachine, Start, Choice, Stop, Action, Call, RootMachine, LoopInvokerMachine, MachineRelation, \
    RandomMachine
from HAM_new.HAM_experiments.HAM_utils import HAMParamsCommon, maze_world_input_01, plot_multi, ham_runner, PlotParams
from environments.arm_env.arm_env import ArmEnv
from environments.grid_maze_env.maze_world_env import MazeWorldEpisodeLength
from utils.graph_drawer import draw_graph

# env = ArmEnv(size_x=3, size_y=3, cubes_cnt=3, episode_max_length=100, finish_reward=100, action_minus_reward=1, tower_target_size=3)
env = MazeWorldEpisodeLength(maze=maze_world_input_01())

machine = RandomMachine(env)
draw_graph("random_machine", machine.get_graph_to_draw())
