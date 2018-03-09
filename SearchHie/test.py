from random import randrange

from HAM.HAM_experiments.HAM_utils import ham_runner, HAMParamsCommon
from article_experiments.global_envs import MazeEnvArticle, MazeEnvArticleSpecial, ArmEnvArticle, EnvironmentsArticle, get_cumulative_rewards
from environments.weak_methods import q_learning

from HAM.HAM_core import AbstractMachine, Action, Stop, MachineRelation, Start, MachineGraph, Choice, Call, RootMachine, LoopInvokerMachine
from SearchHie import goodhams
from SearchHie.main import environments
from environments.arm_env.arm_env import ArmEnvToggleTopOnly
from utils.graph_drawer import draw_graph


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
    # len_ = len(goodhams)
    # print(len_)
    # len_4 = len_ // 4 + 1
    # l1, r1 = 0, len_4
    # l2, r2 = len_4, 2 * len_4
    # l3, r3 = 2 * len_4, 3 * len_4
    # l4, r4 = 3 * len_4, 4 * len_4

    # print(l1, r1 )
    # print(l2, r2 )
    # print(l3, r3 )
    # print(l4, r4 )
    # exit(0)
    # for brute_force in goodhams:
    # for index, brute_force in enumerate(goodhams[l1: r1]):
    # for index, brute_force in enumerate(goodhams[l2: r2]):
    # for index, brute_force in enumerate(goodhams[l3: r3]):
    brute_force = 1180698

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
    transitions.append(MachineRelation(left=Start(), right=a[0]))
    machine = AbstractMachine(MachineGraph(transitions=transitions))
    am = RootMachine(LoopInvokerMachine(machine))
    env = environments[0]
    draw_graph("{brute_force}".format(**locals()), am.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))

    name = "02_auto"

    def run(global_env):
        full_name = name
        params = HAMParamsCommon(environments[0])
        ham_runner(ham=am, num_episodes=global_episodes, env=env,params=params)
        rewards = params.logs["ep_rewards"]
        # with open(full_name + " cumulative_reward.txt", "w") as w:
        #     for out in get_cumulative_rewards(rewards=rewards):
        #         w.write(str(out) + '\n', )

        with open(full_name + " reward.txt", "w") as w:
            for out in rewards:
                w.write(str(out) + '\n', )

    def main():
        # for global_env in EnvironmentsArticle().environments:
        run(EnvironmentsArticle().environments[0])

    if __name__ == '__main__':
        main()


env = environments[3]

global_episodes = 6000


def go_q_learn():
    name = "01_table_q-learning"

    def run(global_env):
        full_name = name
        rewards, _ = q_learning(env=env, num_episodes=global_episodes)

        # with open(full_name + " cumulative_reward.txt", "w") as w:
        #     for out in get_cumulative_rewards(rewards=rewards):
        #         w.write(str(out) + '\n', )

        with open(full_name + " reward.txt", "w") as w:
            for out in rewards:
                w.write(str(out) + '\n', )

    def main():
        # for global_env in EnvironmentsArticle().environments:
        run(EnvironmentsArticle().environments[0])

    if __name__ == '__main__':
        main()


if __name__ == '__main__':
    go_q_learn()
    main()
