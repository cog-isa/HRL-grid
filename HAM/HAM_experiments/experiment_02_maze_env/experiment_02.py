from HAM.HAM_core import AutoBasicMachine, AbstractMachine, Start, Choice, Stop, Action, Call, RootMachine, LoopInvokerMachine, MachineRelation, \
    RandomMachine
from HAM.HAM_experiments.HAM_utils import HAMParamsCommon, maze_world_input_01, plot_multi, ham_runner, PlotParams
from environments.arm_env.arm_env import ArmEnv
from environments.grid_maze_env.maze_world_env import MazeWorldEpisodeLength
from utils.graph_drawer import draw_graph

to_plot = []
# env = ArmEnv(size_x=3, size_y=3, cubes_cnt=3, episode_max_length=100, finish_reward=100, action_minus_reward=1, tower_target_size=3)
for test in range(50):
    print("test:", test)
    env = MazeWorldEpisodeLength(maze=maze_world_input_01())

    for _ in range(5):
        new_machine = RandomMachine().with_new_vertex(env=env)
    for _ in range(15):
        try:
            new_machine = new_machine.with_new_relation()
        except AssertionError:
            pass

    draw_graph(str(test), new_machine.get_graph_to_draw())

    num_episodes = 400

    params = HAMParamsCommon(env)
    try:
        ham_runner(ham=RootMachine(machine_to_invoke=LoopInvokerMachine(new_machine)), num_episodes=num_episodes, env=env, params=params)
        to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="Random" + str(test + 1)))
    except KeyError:
        print("keyError")
    except AssertionError:
        print("assertation")

plot_multi(to_plot)
