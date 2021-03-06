from collections import namedtuple

import sys

from HAM.HAM_core import HAMParams, RandomMachine, LoopInvokerMachine, RootMachine, Start, Choice, Action, Stop, Call, \
    MachineRelation, MachineGraph, AbstractMachine
from environments.grid_maze_env.grid_maze_generator import generate_pattern, generate_maze, place_start_finish, prepare_maze
from utils import plotting
from utils.graph_drawer import draw_graph


class HAMParamsCommon(HAMParams):
    def __init__(self, env):
        super().__init__(q_value={},
                         env=env,
                         current_state=None,
                         eps=0.1,
                         gamma=0.9,
                         alpha=0.1,
                         string_prefix_of_machine=None,
                         accumulated_discount=1,
                         accumulated_rewards=0,
                         previous_machine_choice_state=None,
                         env_is_done=None,
                         logs={"reward": 0, "ep_rewards": []},
                         on_model_transition_id_function=lambda env_: 1 if env_.is_done() else 0, )


def maze_world_input_01():
    base_patterns = [2 ** 4 + 2 ** 8, 1 + 2 ** 12, 0]
    x = list(map(generate_pattern, base_patterns))

    mz_level1 = generate_maze(x, size_x=3, size_y=3, seed=15)
    mz_level2 = generate_maze([mz_level1], size_x=3, size_y=2)
    return place_start_finish(prepare_maze(mz_level2))


def plot_multi(p_params, filename=None):
    plotting.plot_multi_test(smoothing_window=30,
                             x_label="episode",
                             y_label="smoothed rewards",
                             curve_to_draw=[_.curve_to_draw for _ in p_params],
                             labels=[_.label for _ in p_params],
                             filename=filename
                             )


def ham_runner(ham, num_episodes, env, params, no_output=None):
    for i_episode in range(1, num_episodes + 1):
        env.reset()
        ham.run(params)
        assert env.is_done(), "The machine is STOPPED before STOP(done) of the environment"
        if i_episode % 10 == 0:
            if no_output is None:
                print("\r{ham} episode {i_episode}/{num_episodes}.".format(**locals()), end="")
                sys.stdout.flush()


def draw_system_machines():
    s1 = RandomMachine()
    s2 = LoopInvokerMachine(machine_to_invoke=s1)
    s3 = RootMachine(machine_to_invoke=s2)

    draw_graph("full_hie", s3.get_graph_to_draw())


PlotParams = namedtuple("PlotParams", ["curve_to_draw", "label"])


def super_runner(call_me_maybe, env):
    start = Start()
    choice_one = Choice()
    actions = [Action(action=_) for _ in env.get_actions_as_dict().values()]
    stop = Stop()

    call = Call(call_me_maybe)
    transitions = [MachineRelation(left=start, right=choice_one), ]
    for action in actions:
        transitions.append(MachineRelation(left=choice_one, right=action))
        transitions.append(MachineRelation(left=action, right=stop, label=0))
        transitions.append(MachineRelation(left=action, right=stop, label=1))
    transitions.append(MachineRelation(left=choice_one, right=call))
    transitions.append(MachineRelation(left=call, right=stop))

    return AbstractMachine(graph=MachineGraph(transitions=transitions))

def main():
    draw_system_machines()


if __name__ == '__main__':
    main()
