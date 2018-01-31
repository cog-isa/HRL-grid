import random

from HAM.HAM_core import AutoBasicMachine, AbstractMachine, Start, Choice, Stop, Action, Call, RootMachine, LoopInvokerMachine, MachineRelation, \
    RandomMachine, MachineGraph
from HAM.HAM_experiments.HAM_utils import HAMParamsCommon, maze_world_input_01, plot_multi, ham_runner, PlotParams
from environments.arm_env.arm_env import ArmEnv
from environments.grid_maze_env.grid_maze_generator import generate_maze_please, draw_maze
from environments.grid_maze_env.maze_world_env import MazeWorldEpisodeLength
from utils.graph_drawer import draw_graph

to_plot = []


def dfs_check_graph_for_no_action_loops(graph: MachineGraph, current_vertex, visited, ok, action_vertex_was_visited):
    if current_vertex not in visited:
        visited.append(current_vertex)
        if isinstance(current_vertex, Action):
            action_vertex_was_visited = True
        if action_vertex_was_visited:
            ok.append(current_vertex)
        for relation in graph.vertex_mapping[current_vertex]:
            to = relation.right
            dfs_check_graph_for_no_action_loops(graph, to, visited, ok, action_vertex_was_visited)


def get_reachable_vertices_dfs(graph, vertices_to_go_from, reachable=None):
    if reachable is None:
        reachable = set()
    for vertex in vertices_to_go_from:
        if vertex in reachable:
            continue
        reachable.add(vertex)
        for relation in graph.vertex_mapping[vertex]:
            vertex_to_go = relation.right
            RandomMachine.dfs_get_reachable_vertices(graph, [vertex_to_go], reachable)
    return reachable




for test in range(500):
    print('\n', "*******" * 5)
    print("test:{test}".format(**locals()), end="\n\n")

    env = MazeWorldEpisodeLength(generate_maze_please(size_x=2, size_y=2))
    # env = MazeWorldEpisodeLength(maze=maze_world_input_01(), episode_max_length=400, finish_reward=500)
    # draw_maze(generate_maze_please(size_x=2, size_y=2))
    num_episodes = 800
    number_of_vertex = random.randrange(1, 8)
    number_of_edges = random.randrange(1, 12)
    new_machine = RandomMachine().with_new_vertex(env=env)
    for _ in range(number_of_vertex):
        new_machine = new_machine.with_new_vertex(env=env)
    for _ in range(number_of_edges):
        try:
            new_machine = new_machine.with_new_relation()
        except AssertionError:
            pass
    ok = []
    dfs_check_graph_for_no_action_loops(graph=new_machine.graph, current_vertex=new_machine.graph.get_start(), visited=[], ok=ok,
                                        action_vertex_was_visited=False)
    # TODO filter graphs with loops on Choice vertex
    # graph is consider as correct:
    # first - if we can reach each vertex from Start
    # second - if we can reach each vertex from Stop, going over edges with reversed order

    if new_machine.graph.get_stop() in ok:



        params = HAMParamsCommon(env)
        try:
            ham_runner(ham=RootMachine(machine_to_invoke=LoopInvokerMachine(new_machine)), num_episodes=num_episodes, env=env, params=params)
            to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="Random" + str(test + 1)))
            if sum(params.logs["ep_rewards"][-100:]) > 0:
                draw_graph("pics/" + str(test), new_machine.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))
            print("\n\nsum:", sum(params.logs["ep_rewards"]))
        except KeyError:
            print("keyError")
        except AssertionError:
            print("assertion")

plot_multi(to_plot)
