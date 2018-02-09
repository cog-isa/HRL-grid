import random

from HAM.HAM_core import Action, RootMachine, \
    LoopInvokerMachine, RandomMachine, MachineGraph
from HAM.HAM_experiments.HAM_utils import HAMParamsCommon, plot_multi, ham_runner, PlotParams, super_runner
from article_experiments.global_envs import MazeEnvArticleSpecial, MazeEnvArticle, ArmEnvArticle
from environments.grid_maze_env.grid_maze_generator import generate_pattern, generate_maze, place_start_finish, \
    prepare_maze, generate_maze_please
from environments.grid_maze_env.maze_world_env import MazeWorldEpisodeLength
from utils.graph_drawer import draw_graph

to_plot = []


def dfs_check_graph_for_no_action_loops(graph: MachineGraph, current_vertex, visited, ok, action_vertex_was_visited):
    if current_vertex not in visited or (action_vertex_was_visited and current_vertex not in ok):
        visited.append(current_vertex)
        if isinstance(current_vertex, Action):
            action_vertex_was_visited = True
        if action_vertex_was_visited:
            ok.append(current_vertex)
        for relation in graph.vertex_mapping[current_vertex]:
            to = relation.right
            dfs_check_graph_for_no_action_loops(graph, to, visited, ok, action_vertex_was_visited)


def dfs_distinct_from_start(graph: MachineGraph, vertex, visited, reversed_order=None):
    if vertex in visited:
        return visited
    visited.append(vertex)
    mapping = graph.vertex_mapping if reversed_order is None else graph.vertex_reverse_mapping
    for relation in mapping[vertex]:
        to = relation.right if reversed_order is None else relation.left
        dfs_distinct_from_start(graph=graph, vertex=to, visited=visited, reversed_order=reversed_order)
    return visited


def is_it_machine_runnable(machine):
    ok = []
    dfs_check_graph_for_no_action_loops(graph=machine.graph, current_vertex=machine.graph.get_start(),
                                        visited=[], ok=ok,
                                        action_vertex_was_visited=False)
    if machine.graph.get_stop() not in ok:
        return False

    x = dfs_distinct_from_start(graph=machine.graph, vertex=machine.graph.get_start(), visited=[])
    y = dfs_distinct_from_start(graph=machine.graph, vertex=machine.graph.get_stop(), visited=[],
                                reversed_order=True)

    if set(x) != set(y):
        return False
    return True


def create_random_machine(maximal_number_of_vertex, maximal_number_of_edges, random_seed, env):
    random.seed(random_seed)
    number_of_vertex = random.randrange(1, maximal_number_of_vertex)
    number_of_edges = random.randrange(1, maximal_number_of_edges)
    new_machine = RandomMachine().with_new_vertex(env=env)
    for _ in range(number_of_vertex):
        new_machine = new_machine.with_new_vertex(env=env)
    for __ in range(number_of_edges):
        try:
            new_machine = new_machine.with_new_relation()
        except AssertionError:
            break
    return new_machine


def main(begin_seed=0):
    for seed in range(begin_seed, begin_seed + 5000):
        # maze = maze_world_input_special()
        # maze = generate_maze_please(size_x=2, size_y=2)
        # env = MazeWorldEpisodeLength(maze=maze)
        # global_env, save_folder  = MazeEnvArticleSpecial(), "laby_spec/"
        global_env, save_folder  = MazeEnvArticle(), "laby/"
        # global_env, save_folder  = ArmEnvArticle(), "arm/"

        env, num_episodes = global_env.env, global_env.episodes_count

        new_machine = create_random_machine(maximal_number_of_vertex=6, maximal_number_of_edges=6, random_seed=seed,
                                            env=env)

        if is_it_machine_runnable(new_machine):
            params = HAMParamsCommon(env)
            try:
                ham_runner(
                    ham=RootMachine(LoopInvokerMachine(machine_to_invoke=super_runner(new_machine, env))),
                    num_episodes=num_episodes,
                    env=env, params=params,
                    no_output=True
                    )
                ham_runner(ham=RootMachine(machine_to_invoke=LoopInvokerMachine(new_machine)),
                           num_episodes=num_episodes,
                           env=env, params=params, no_output=True)

                # to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="Random" + str(seed + 1)))
                reward = sum(params.logs["ep_rewards"])
                draw_graph(save_folder + str(reward) + ":::" + str(seed),
                           new_machine.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))
                # draw_graph("pics/" + str(reward).rjust(10, "0"),
                #            new_machine.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))
            except KeyError:
                print("keyError", end="")
            except AssertionError:
                print("assertion", end="")
    plot_multi(to_plot)


if __name__ == '__main__':
    main(begin_seed=random.randrange(1, 2000000000))
