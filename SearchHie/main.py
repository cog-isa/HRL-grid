from random import randrange

from HAM.HAM_core import Action, Choice, Call, AbstractMachine, MachineGraph, MachineRelation, Start, Stop, RandomMachine, RootMachine, LoopInvokerMachine
from HAM.HAM_experiments.HAM_utils import ham_runner, HAMParamsCommon
from HAM.HAM_experiments.experiment_04_auto_random_HAM_on_maze_env.experiment_04 import is_it_machine_runnable
from SearchHie.goodhams import goodhams
from environments.arm_env.arm_env import ArmEnvToggleTopOnly
from utils.graph_drawer import draw_graph

default_size_x = 7
default_episode_max_length = 400
default_finish_reward = 1
default_action_minus_reward = -0.02

environments = [
    ArmEnvToggleTopOnly(
        size_y=2,
        cubes_cnt=2,
        tower_target_size=2,
        size_x=default_size_x, episode_max_length=default_episode_max_length, finish_reward=default_finish_reward,
        action_minus_reward=default_action_minus_reward),

    ArmEnvToggleTopOnly(
        size_y=3,
        cubes_cnt=3,
        tower_target_size=3,
        size_x=default_size_x, episode_max_length=default_episode_max_length, finish_reward=default_finish_reward,
        action_minus_reward=default_action_minus_reward),

    ArmEnvToggleTopOnly(
        size_y=4,
        cubes_cnt=4,
        tower_target_size=3,
        size_x=default_size_x, episode_max_length=default_episode_max_length, finish_reward=default_finish_reward,
        action_minus_reward=default_action_minus_reward),

    ArmEnvToggleTopOnly(
        size_y=5,
        cubes_cnt=5,
        tower_target_size=4,
        size_x=default_size_x, episode_max_length=default_episode_max_length, finish_reward=default_finish_reward,
        action_minus_reward=default_action_minus_reward)
]

env = environments[0]


def go(transitions, brute_force, index_):
    machine = AbstractMachine(MachineGraph(transitions=transitions))
    am = RootMachine(LoopInvokerMachine(machine))

    # if randrange(1000) == 0:
    #     draw_graph("{brute_force}".format(**locals()), am.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))
    #     exit(0)

    if is_it_machine_runnable(machine):
        sum_rew = 0
        try:
            params = HAMParamsCommon(environments[0])
            ham_runner(ham=am, num_episodes=2, env=environments[0], params=params)
            sum_rew = sum(params.logs["ep_rewards"])


        except ChildProcessError:
            # print(brute_force)
            pass
            # if randrange(1500) == 0:
            #     draw_graph("bf{brute_force}".format(**locals()), am.get_graph_to_draw())

        if sum_rew > 0:
            # TODO
            # with open("out.txt", "a") as f:
            #     f.write(str(brute_force) + "\n")
            # return

            # print("\n\n EPISODE REWARD: ", sum_rew)
            # draw_graph("{sum_rew}__{brute_force}".format(**locals()), am.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))
            rew = None
            print("\n\n\n")
            for e in environments:
                params = HAMParamsCommon(e)
                ham_runner(ham=am, num_episodes=600, env=e, params=params)
                if rew is None:
                    rew = 0
                rew += sum(params.logs["ep_rewards"])
                print("to_add:", sum(params.logs["ep_rewards"]))
                # except ChildProcessError:
                #     draw_graph("{rew}__{brute_force}".format(**locals()), am.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))
                #     exit(0)
                # pass
            if rew is not None:
                draw_graph("{rew}__{brute_force}_{index_}".format(**locals()), am.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))


def main():
    class UpMachine4(AbstractMachine):
        def __init__(self, env: ArmEnvToggleTopOnly):
            d1 = Action(action=env.ACTIONS.UP)
            d2 = Action(action=env.ACTIONS.UP)
            d3 = Action(action=env.ACTIONS.UP)
            d4 = Action(action=env.ACTIONS.UP)
            stop = Stop()
            transitions = (
                MachineRelation(left=Start(), right=d1),
                MachineRelation(left=d1, right=d2, label=0),
                MachineRelation(left=d2, right=d3, label=0),
                MachineRelation(left=d3, right=d4, label=0),
                MachineRelation(left=d4, right=stop, label=0),

                MachineRelation(left=d1, right=stop, label=1),
                MachineRelation(left=d2, right=stop, label=1),
                MachineRelation(left=d3, right=stop, label=1),
                MachineRelation(left=d4, right=stop, label=1),

            )

            super().__init__(graph=MachineGraph(transitions=transitions))

    class UpMachine3(AbstractMachine):
        def __init__(self, env: ArmEnvToggleTopOnly):
            d1 = Action(action=env.ACTIONS.UP)
            d2 = Action(action=env.ACTIONS.UP)
            d3 = Action(action=env.ACTIONS.UP)
            # d4 = Action(action=env.ACTIONS.UP)
            stop = Stop()
            transitions = (
                MachineRelation(left=Start(), right=d1),
                MachineRelation(left=d1, right=d2, label=0),
                MachineRelation(left=d2, right=d3, label=0),
                MachineRelation(left=d3, right=stop, label=0),
                # MachineRelation(left=d4, right=stop, label=0),

                MachineRelation(left=d1, right=stop, label=1),
                MachineRelation(left=d2, right=stop, label=1),
                MachineRelation(left=d3, right=stop, label=1),
                # MachineRelation(left=d4, right=stop, label=1),

            )

            super().__init__(graph=MachineGraph(transitions=transitions))

    a = [
        Choice(),
        Action(ArmEnvToggleTopOnly.ACTIONS.RIGHT),
        Action(ArmEnvToggleTopOnly.ACTIONS.LEFT),
        Action(ArmEnvToggleTopOnly.ACTIONS.DOWN),
        # Action(ArmEnvToggleTopOnly.ACTIONS.UP),

        Call(machine_to_call=UpMachine4(environments[1])),

    ]

    transitions = []
    for i in a:
        for j in a:
            if randrange(2):
                if isinstance(i, Action):
                    transitions.append(MachineRelation(left=i, right=j, label=0))
                else:
                    transitions.append(MachineRelation(left=i, right=j))
    len_ = len(goodhams)
    print(len_)
    len_4 = len_ // 4 + 1
    l1, r1 = 0, len_4
    l2, r2 = len_4, 2 * len_4
    l3, r3 = 2 * len_4, 3 * len_4
    l4, r4 = 3 * len_4, 4 * len_4

    # print(l1, r1 )
    # print(l2, r2 )
    # print(l3, r3 )
    # print(l4, r4 )
    # exit(0)
    # for brute_force in goodhams:
    # for index, brute_force in enumerate(goodhams[l1: r1]):
    # for index, brute_force in enumerate(goodhams[l2: r2]):
    # for index, brute_force in enumerate(goodhams[l3: r3]):
    for index, brute_force in enumerate(goodhams[l4: r4]):
        if index >= len_:
            break
        if index % (len_ // 100) == 0:
            print(index // (len_ // 100), "%")

        if bin(brute_force).count("1") > 10:
            continue

        if bin(brute_force).count("1") < 4:
            continue

        # if bin(brute_force).count("1") > 12 or bin(brute_force).count("1") < 4:
        #     continue

        # continue
        go_continue = False
        transitions = []
        ss = set()
        for ii in range(len(a)):
            for jj in range(len(a)):
                i = a[ii]
                j = a[jj]
                if (2 ** (ii * len(a) + jj)) & brute_force:
                    if isinstance(i, Action):
                        transitions.append(MachineRelation(left=i, right=j, label=0))
                    else:
                        transitions.append(MachineRelation(left=i, right=j))
                    if ii in ss and isinstance(a[ii], (Action, Call)):
                        go_continue = True
                        break
                    ss.add(ii)
        if go_continue:
            # print('continue')
            continue
        stop = Stop()
        for ii in range(len(a)):
            if ii not in ss:
                i = a[ii]
                if isinstance(i, Action):
                    transitions.append(MachineRelation(left=i, right=stop, label=0))
                else:
                    transitions.append(MachineRelation(left=i, right=stop))
        for i in a:
            if isinstance(i, Action):
                transitions.append(MachineRelation(left=i, right=stop, label=1))

        for index_, II in enumerate(a):
            transitions.append(MachineRelation(left=Start(), right=II))
            go(transitions=transitions, brute_force=brute_force, index_=index_)
            transitions.pop()


if __name__ == '__main__':
    main()
