import pathlib

from HAM.HAM_core import Stop, Start, Action, Call, Choice, AbstractMachine, MachineRelation
from HAM.HAM_experiments.experiment_04_auto_random_HAM_on_maze_env.experiment_04 import is_it_machine_runnable, dfs_distinct_from_start
from environments.arm_env.arm_env import ArmEnvToggleTopOnly
from workshop.generate_combination import vertex_combination, vertex_list_to_str
from workshop.generate_graph import MachineStored

import shutil


def is_ham_ok(machine: AbstractMachine):
    start = machine.graph.get_start()
    # check for exact 1 outgoing edges from Start
    if len(machine.graph.vertex_mapping[start]) != 1:
        return False
    # check for no incoming edges to Start
    if len(machine.graph.vertex_reverse_mapping[start]) != 0:
        return False

    stop = machine.graph.get_stop()
    # check for no outgoing edges from Stop
    if len(machine.graph.vertex_mapping[stop]) != 0:
        return False

    # check for exact 2 outgoing edges from actions (1 vertex is on_model end to stop)
    for action in machine.graph.get_special_vertices(Action):
        if len(machine.graph.vertex_mapping[action]) != 1:
            return False

    # check for self loops
    for edge in machine.graph.transitions:
        # MachineRelation.left
        if edge.left is edge.right:
            return False

    # no edges from Choice to Stop
    for choice in machine.graph.get_special_vertices(Choice):
        for relation in machine.graph.vertex_mapping[choice]:
            if isinstance(relation.right, Stop):
                return False

    # check for more than 1 outgoing edges from Choice
    for choice in machine.graph.get_special_vertices(Choice):
        if len(machine.graph.vertex_mapping[choice]) <= 1:
            return False

    return True


def check_for_one_component_graph(machine: AbstractMachine):
    visited = dfs_distinct_from_start(graph=machine.graph, vertex=machine.graph.get_start(), visited=[])
    return len(visited) == len(machine.graph.vertices)


def get_graph_id_fast(m: MachineStored, current_index=0, cur_id=0, ans=None):
    if ans is None:
        ans = []
    if current_index == len(m.vertex_types):
        ans.append(cur_id)
        return ans

    if isinstance(m.vertex_types[current_index], (Action, Start)):
        for i in range(1, len(m.vertex_types)):
            if i == current_index:
                continue
            get_graph_id_fast(m, current_index + 1, cur_id + (2 ** i) * (2 ** (len(m.vertex_types * current_index))), ans=ans)
    elif isinstance(m.vertex_types[current_index], Stop):
        get_graph_id_fast(m, current_index + 1, cur_id, ans=ans)
    elif isinstance(m.vertex_types[current_index], Choice):
        for i in range(1, 2 ** len(m.vertex_types)):
            get_graph_id_fast(m, current_index + 1, cur_id + i * (2 ** (len(m.vertex_types * current_index))), ans=ans)
    else:
        raise TypeError
    return ans


def generate_good_graphs(env, vertexes, vertex_count):
    good_graphs = []
    vertex_count += 1
    for max_vertex_count in range(vertex_count):
        vc = vertex_combination(vertex_types=vertexes, max_vertex_count=max_vertex_count)
        for index, vertex_types in enumerate(vc):
            for graph_id in sorted(get_graph_id_fast(MachineStored(vertex_types=vertex_types, binary_matrix_representation=412, env=env))):
                ms = MachineStored(vertex_types=vertex_types, binary_matrix_representation=graph_id, env=env)
                if is_ham_ok(ms.get_machine_without_on_model()):
                    if check_for_one_component_graph(ms.get_machine_without_on_model()):
                        if is_it_machine_runnable(ms.get_machine_without_on_model()):
                            good_graphs.append(ms)
    return good_graphs


def generate_good_graph_ids(env, vertexes, vertex_count):
    good_graphs = []
    vertex_count += 1
    for max_vertex_count in range(vertex_count):
        vc = vertex_combination(vertex_types=vertexes, max_vertex_count=max_vertex_count)
        for index, vertex_types in enumerate(vc):
            for graph_id in sorted(get_graph_id_fast(MachineStored(vertex_types=vertex_types, binary_matrix_representation=412, env=env))):
                ms = MachineStored(vertex_types=vertex_types, binary_matrix_representation=graph_id, env=env)
                if is_ham_ok(ms.get_machine_without_on_model()):
                    if check_for_one_component_graph(ms.get_machine_without_on_model()):
                        if is_it_machine_runnable(ms.get_machine_without_on_model()):
                            good_graphs.append(graph_id)
    return good_graphs


def generate_machines_by_ids(env, vertexes, ids):
    machines = []
    for max_vertex_count in range(7):
        vc = vertex_combination(vertex_types=vertexes, max_vertex_count=max_vertex_count)
        for index, vertex_types in enumerate(vc):
            for graph_id in ids:
                ms = MachineStored(vertex_types=vertex_types, binary_matrix_representation=graph_id, env=env)
                if is_ham_ok(ms.get_machine_without_on_model()):
                    if check_for_one_component_graph(ms.get_machine_without_on_model()):
                        if is_it_machine_runnable(ms.get_machine_without_on_model()):
                            machines.append(ms)
    return machines


def main():
    env = ArmEnvToggleTopOnly(size_x=5, size_y=5, cubes_cnt=4, episode_max_length=600, finish_reward=100, action_minus_reward=-0.001, tower_target_size=4)
    vertexes = sorted([
        Stop(),
        Start(),

        Action(env.ACTIONS.LEFT),
        Action(env.ACTIONS.RIGHT),
        Action(env.ACTIONS.UP),
        Action(env.ACTIONS.DOWN),
        # Action(env.ACTIONS.TOGGLE),
        Choice(),
        # Action(env.ACTIONS.LEFT),
        # Action(env.ACTIONS.RIGHT),
        # Action(env.ACTIONS.UP),
        # Action(env.ACTIONS.DOWN),
        # Action(env.ACTIONS.TOGGLE),
        # Choice(),
    ])

    # clearing directory
    pathlib.Path('pics/').mkdir(parents=True, exist_ok=True)
    shutil.rmtree('pics/')
    pathlib.Path('pics/').mkdir(parents=True, exist_ok=True)
    # brute force
    for max_vertex_count in range(7):
        vc = vertex_combination(vertex_types=vertexes, max_vertex_count=max_vertex_count)
        for index, vertex_types in enumerate(vc):
            for graph_id in sorted(get_graph_id_fast(MachineStored(vertex_types=vertex_types, binary_matrix_representation=412, env=env))):
                ms = MachineStored(vertex_types=vertex_types, binary_matrix_representation=graph_id, env=env)
                if is_ham_ok(ms.get_machine_without_on_model()):
                    if check_for_one_component_graph(ms.get_machine_without_on_model()):
                        if is_it_machine_runnable(ms.get_machine_without_on_model()):
                            ms.draw("pics/" + str(max_vertex_count) + ":" + str(index) + ":" + str(graph_id))
                            print("added")


if __name__ == '__main__':
    main()
