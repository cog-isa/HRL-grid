from HAM.HAM_core import Stop, Start, Action, Call, Choice, AbstractMachine
from HAM.HAM_experiments.experiment_04_auto_random_HAM_on_maze_env.experiment_04 import is_it_machine_runnable
from environments.arm_env.arm_env import ArmEnvToggleTopOnly
from workshop.generate_combination import vertex_combination, vertex_list_to_str
from workshop.generate_graph import MachineStored


def is_ham_ok(machine: AbstractMachine):
    # check for only 1 outgoing edges from Start
    raise NotImplementedError
    # check for no incoming edges to Start

    print(*machine.graph.transitions)
    exit(0)


def main():
    env = ArmEnvToggleTopOnly(size_x=5, size_y=5, cubes_cnt=4, episode_max_length=600, finish_reward=100, action_minus_reward=-0.001, tower_target_size=4)
    vertex_types = sorted([
        Stop(),
        Start(),

        Action(env.ACTIONS.LEFT),
        Action(env.ACTIONS.RIGHT),
        Action(env.ACTIONS.UP),
        Action(env.ACTIONS.DOWN),
        Action(env.ACTIONS.TOGGLE),
        Choice(),
        Action(env.ACTIONS.LEFT),
        Action(env.ACTIONS.RIGHT),
        Action(env.ACTIONS.UP),
        Action(env.ACTIONS.DOWN),
        Action(env.ACTIONS.TOGGLE),
        # Choice(),
    ])

    for index, vertex_types in enumerate(vertex_combination(vertex_types=vertex_types, max_vertex_count=5)):
        for graph_id in range(MachineStored(vertex_types=vertex_types, binary_matrix_representation=412, env=env).get_max_index()):
            ms = MachineStored(vertex_types=vertex_types, binary_matrix_representation=graph_id, env=env)
            if is_ham_ok(ms.get_machine()):
                if is_it_machine_runnable(ms.get_machine()):
                    ms.draw("pics/" + str(graph_id))
        exit(0)


if __name__ == '__main__':
    main()
