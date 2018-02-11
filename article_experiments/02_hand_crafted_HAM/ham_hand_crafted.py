from HAM.HAM_core import AbstractMachine, Start, Action, Stop, MachineRelation, MachineGraph, Choice, Call, RootMachine, LoopInvokerMachine
from HAM.HAM_experiments.HAM_utils import ham_runner, HAMParamsCommon
from article_experiments.global_envs import MazeEnvArticle, MazeEnvArticleSpecial, ArmEnvArticle, EnvironmentsArticle, get_cumulative_rewards
from environments.grid_maze_env.maze_world_env import MazeWorldEpisodeLength
from utils.graph_drawer import draw_graph

name = "02_ham_hand_crafted"


def run(global_env):
    if isinstance(global_env, ArmEnvArticle):
        env = global_env.env
        internal_machine = PullUpMachine(env=env)
        machine = RootMachine(machine_to_invoke=LoopInvokerMachine(machine_to_invoke=internal_machine))
        params = HAMParamsCommon(env)
        draw_graph(file_name="arm_env", graph=internal_machine.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))
        ham_runner(ham=machine, num_episodes=global_env.episodes_count, env=env, params=params)
        rewards = params.logs["ep_rewards"]

    elif isinstance(global_env, MazeEnvArticle):
        env = global_env.env
        internal_machine = InterestingMachine(env=env)
        machine = RootMachine(machine_to_invoke=LoopInvokerMachine(machine_to_invoke=internal_machine))
        draw_graph(file_name="maze_env",graph=internal_machine.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))
        params = HAMParamsCommon(env)
        ham_runner(ham=machine, num_episodes=global_env.episodes_count, env=env, params=params)
        rewards = params.logs["ep_rewards"]
    elif isinstance(global_env, MazeEnvArticleSpecial):
        env = global_env.env
        internal_machine = InterestingMachineLeftUpInteresting(env=env)
        machine = RootMachine(machine_to_invoke=LoopInvokerMachine(machine_to_invoke=internal_machine))
        draw_graph(file_name="maze_env_special",graph=internal_machine.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))
        params = HAMParamsCommon(env)
        ham_runner(ham=machine, num_episodes=global_env.episodes_count, env=env, params=params)
        rewards = params.logs["ep_rewards"]
    else:
        raise KeyError
    full_name = name + "_" + global_env.__class__.__name__
    # with open(full_name + " cumulative_reward.txt", "w") as w:
    #     for out in get_cumulative_rewards(rewards=rewards):
    #         w.write(str(out) + '\n', )

    with open(full_name + " reward.txt", "w") as w:
        for out in rewards:
            w.write(str(out) + '\n', )


class PullUpMachine(AbstractMachine):
    def __init__(self, env):
        pull_up_start = Start()
        pull_up_on = Action(action=env.get_actions_as_dict()["ON"])
        pull_up_down_01 = Action(action=env.get_actions_as_dict()["DOWN"])
        pull_up_down_02 = Action(action=env.get_actions_as_dict()["DOWN"])
        pull_up_down_03 = Action(action=env.get_actions_as_dict()["DOWN"])
        pull_up_down_04 = Action(action=env.get_actions_as_dict()["DOWN"])
        pull_up_up_01 = Action(action=env.get_actions_as_dict()["UP"])
        pull_up_up_02 = Action(action=env.get_actions_as_dict()["UP"])
        pull_up_up_03 = Action(action=env.get_actions_as_dict()["UP"])
        pull_up_up_04 = Action(action=env.get_actions_as_dict()["UP"])
        pull_up_stop = Stop()

        pull_up_transitions = (
            MachineRelation(left=pull_up_start, right=pull_up_on),

            MachineRelation(left=pull_up_on, right=pull_up_down_01, label=0),
            MachineRelation(left=pull_up_down_01, right=pull_up_down_02, label=0),
            MachineRelation(left=pull_up_down_02, right=pull_up_down_03, label=0),
            MachineRelation(left=pull_up_down_03, right=pull_up_down_04, label=0),
            MachineRelation(left=pull_up_down_04, right=pull_up_up_01, label=0),
            MachineRelation(left=pull_up_up_01, right=pull_up_up_02, label=0),
            MachineRelation(left=pull_up_up_02, right=pull_up_up_03, label=0),
            MachineRelation(left=pull_up_up_03, right=pull_up_up_04, label=0),
            MachineRelation(left=pull_up_up_04, right=pull_up_stop, label=0),

            MachineRelation(left=pull_up_on, right=pull_up_stop, label=1),
            MachineRelation(left=pull_up_down_01, right=pull_up_stop, label=1),
            MachineRelation(left=pull_up_down_02, right=pull_up_stop, label=1),
            MachineRelation(left=pull_up_down_03, right=pull_up_stop, label=1),
            MachineRelation(left=pull_up_down_04, right=pull_up_stop, label=1),
            MachineRelation(left=pull_up_up_01, right=pull_up_stop, label=1),
            MachineRelation(left=pull_up_up_02, right=pull_up_stop, label=1),
            MachineRelation(left=pull_up_up_03, right=pull_up_stop, label=1),
            MachineRelation(left=pull_up_up_04, right=pull_up_stop, label=1),
        )
        pull_up = AbstractMachine(MachineGraph(transitions=pull_up_transitions))

        start = Start()
        choice_one = Choice()
        left = Action(action=env.get_actions_as_dict()["LEFT"])
        right = Action(action=env.get_actions_as_dict()["RIGHT"])
        off = Action(action=env.get_actions_as_dict()["OFF"])

        call = Call(machine_to_call=pull_up)

        stop = Stop()

        transitions = (
            MachineRelation(left=start, right=choice_one),
            MachineRelation(left=choice_one, right=left),
            MachineRelation(left=choice_one, right=right),
            MachineRelation(left=choice_one, right=off),
            MachineRelation(left=choice_one, right=call),

            MachineRelation(left=call, right=stop),

            MachineRelation(left=left, right=stop, label=0),
            MachineRelation(left=right, right=stop, label=0),
            MachineRelation(left=off, right=stop, label=0),

            MachineRelation(left=left, right=stop, label=1),
            MachineRelation(left=right, right=stop, label=1),
            MachineRelation(left=off, right=stop, label=1),
        )

        super().__init__(graph=MachineGraph(transitions=transitions))


class InterestingMachine(AbstractMachine):
    def __init__(self, env: MazeWorldEpisodeLength):
        left1 = Action(action=env.ACTIONS.LEFT)
        left2 = Action(action=env.ACTIONS.LEFT)
        left3 = Action(action=env.ACTIONS.LEFT)
        left4 = Action(action=env.ACTIONS.LEFT)
        left5 = Action(action=env.ACTIONS.LEFT)

        right1 = Action(action=env.ACTIONS.RIGHT)
        right2 = Action(action=env.ACTIONS.RIGHT)
        right3 = Action(action=env.ACTIONS.RIGHT)
        right4 = Action(action=env.ACTIONS.RIGHT)
        right5 = Action(action=env.ACTIONS.RIGHT)

        up1 = Action(action=env.ACTIONS.UP)
        up2 = Action(action=env.ACTIONS.UP)
        up3 = Action(action=env.ACTIONS.UP)
        up4 = Action(action=env.ACTIONS.UP)
        up5 = Action(action=env.ACTIONS.UP)

        down1 = Action(action=env.ACTIONS.DOWN)
        down2 = Action(action=env.ACTIONS.DOWN)
        down3 = Action(action=env.ACTIONS.DOWN)
        down4 = Action(action=env.ACTIONS.DOWN)
        down5 = Action(action=env.ACTIONS.DOWN)

        choice1 = Choice()
        choice2 = Choice()

        left = Action(action=env.ACTIONS.LEFT)
        right = Action(action=env.ACTIONS.RIGHT)
        up = Action(action=env.ACTIONS.UP)
        down = Action(action=env.ACTIONS.DOWN)

        stop = Stop()

        transitions = (
            MachineRelation(left=Start(), right=choice1),

            MachineRelation(left=choice1, right=left1),
            MachineRelation(left=left1, right=left2, label=0),
            MachineRelation(left=left2, right=left3, label=0),
            MachineRelation(left=left3, right=left4, label=0),
            MachineRelation(left=left4, right=left5, label=0),
            MachineRelation(left=left5, right=choice2, label=0),

            MachineRelation(left=left1, right=stop, label=1),
            MachineRelation(left=left2, right=stop, label=1),
            MachineRelation(left=left3, right=stop, label=1),
            MachineRelation(left=left4, right=stop, label=1),
            MachineRelation(left=left5, right=stop, label=1),

            MachineRelation(left=choice1, right=right1),
            MachineRelation(left=right1, right=right2, label=0),
            MachineRelation(left=right2, right=right3, label=0),
            MachineRelation(left=right3, right=right4, label=0),
            MachineRelation(left=right4, right=right5, label=0),
            MachineRelation(left=right5, right=choice2, label=0),

            MachineRelation(left=right1, right=stop, label=1),
            MachineRelation(left=right2, right=stop, label=1),
            MachineRelation(left=right3, right=stop, label=1),
            MachineRelation(left=right4, right=stop, label=1),
            MachineRelation(left=right5, right=stop, label=1),

            MachineRelation(left=choice1, right=up1),

            MachineRelation(left=up1, right=up2, label=0),
            MachineRelation(left=up2, right=up3, label=0),
            MachineRelation(left=up3, right=up4, label=0),
            MachineRelation(left=up4, right=up5, label=0),
            MachineRelation(left=up5, right=choice2, label=0),
            MachineRelation(left=up1, right=stop, label=1),
            MachineRelation(left=up2, right=stop, label=1),
            MachineRelation(left=up3, right=stop, label=1),
            MachineRelation(left=up4, right=stop, label=1),
            MachineRelation(left=up5, right=stop, label=1),

            MachineRelation(left=choice1, right=down1),
            MachineRelation(left=down1, right=down2, label=0),
            MachineRelation(left=down2, right=down3, label=0),
            MachineRelation(left=down3, right=down4, label=0),
            MachineRelation(left=down4, right=down5, label=0),
            MachineRelation(left=down5, right=choice2, label=0),
            MachineRelation(left=down1, right=stop, label=1),
            MachineRelation(left=down2, right=stop, label=1),
            MachineRelation(left=down3, right=stop, label=1),
            MachineRelation(left=down4, right=stop, label=1),
            MachineRelation(left=down5, right=stop, label=1),

            MachineRelation(left=choice2, right=left),
            MachineRelation(left=choice2, right=right),
            MachineRelation(left=choice2, right=up),
            MachineRelation(left=choice2, right=down),

            MachineRelation(left=left, right=stop, label=1, ),
            MachineRelation(left=right, right=stop, label=1),
            MachineRelation(left=up, right=stop, label=1),
            MachineRelation(left=down, right=stop, label=1),

            MachineRelation(left=left, right=stop, label=0, ),
            MachineRelation(left=right, right=stop, label=0),
            MachineRelation(left=up, right=stop, label=0),
            MachineRelation(left=down, right=stop, label=0),

        )

        super().__init__(graph=MachineGraph(transitions=transitions))


class InterestingMachineLeftUpInteresting(AbstractMachine):
    def __init__(self, env: MazeWorldEpisodeLength):
        left4 = Action(action=env.ACTIONS.LEFT)
        left5 = Action(action=env.ACTIONS.LEFT)

        up4 = Action(action=env.ACTIONS.UP)
        up5 = Action(action=env.ACTIONS.UP)

        choice1 = Choice()
        choice2 = Choice()

        left = Action(action=env.ACTIONS.LEFT)
        right = Action(action=env.ACTIONS.RIGHT)
        up = Action(action=env.ACTIONS.UP)
        down = Action(action=env.ACTIONS.DOWN)

        stop = Stop()

        transitions = (
            MachineRelation(left=Start(), right=choice1),

            MachineRelation(left=choice1, right=left4),
            MachineRelation(left=left4, right=left5, label=0),
            MachineRelation(left=left5, right=choice2, label=0),

            MachineRelation(left=left4, right=stop, label=1),
            MachineRelation(left=left5, right=stop, label=1),

            MachineRelation(left=choice1, right=up4),

            MachineRelation(left=up4, right=up5, label=0),
            MachineRelation(left=up5, right=choice2, label=0),
            MachineRelation(left=up4, right=stop, label=1),
            MachineRelation(left=up5, right=stop, label=1),

            MachineRelation(left=choice2, right=left),
            MachineRelation(left=choice2, right=right),
            MachineRelation(left=choice2, right=up),
            MachineRelation(left=choice2, right=down),

            MachineRelation(left=left, right=stop, label=1, ),
            MachineRelation(left=right, right=stop, label=1),
            MachineRelation(left=up, right=stop, label=1),
            MachineRelation(left=down, right=stop, label=1),

            MachineRelation(left=left, right=stop, label=0, ),
            MachineRelation(left=right, right=stop, label=0),
            MachineRelation(left=up, right=stop, label=0),
            MachineRelation(left=down, right=stop, label=0),

        )

        super().__init__(graph=MachineGraph(transitions=transitions))


def main():
    for global_env in EnvironmentsArticle().environments:
        run(global_env)


if __name__ == '__main__':
    main()
