from HAM.HAM_core import RootMachine, LoopInvokerMachine, Action, Choice, MachineRelation, Stop, Start, MachineGraph, \
    AbstractMachine, RandomMachine
from HAM.HAM_experiments.HAM_utils import HAMParamsCommon, ham_runner, super_runner
from HAM.HAM_experiments.experiment_04_auto_random_HAM_on_maze_env.experiment_04 import create_random_machine
from article_experiments.global_envs import MazeEnvArticleSpecial, ArmEnvArticle, MazeEnvArticle, \
    get_cumulative_rewards, EnvironmentsArticle
from environments.grid_maze_env.maze_world_env import MazeWorldEpisodeLength
from utils.graph_drawer import draw_graph

name = "06_net_handcrafted"


def run(global_env):
    rewards = None
    if isinstance(global_env, ArmEnvArticle):
        env = global_env.env
        internal_machine = M1(env=env)
        machine = RootMachine(LoopInvokerMachine(super_runner(call_me_maybe=internal_machine, env=env)))
        draw_graph(file_name="1",
                   graph=internal_machine.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))
        params = HAMParamsCommon(env)
        ham_runner(ham=machine, num_episodes=global_env.episodes_count, env=env, params=params)
        rewards = params.logs["ep_rewards"]
    elif isinstance(global_env, MazeEnvArticle):
        env = global_env.env
        internal_machine = M2(env=env)
        machine = RootMachine(LoopInvokerMachine(super_runner(call_me_maybe=internal_machine, env=env)))
        draw_graph(file_name="2",
                   graph=internal_machine.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))
        params = HAMParamsCommon(env)
        ham_runner(ham=machine, num_episodes=global_env.episodes_count, env=env, params=params)
        rewards = params.logs["ep_rewards"]
    elif isinstance(global_env, MazeEnvArticleSpecial):
        env = global_env.env
        internal_machine = M3(env=env)
        machine = RootMachine(LoopInvokerMachine(super_runner(call_me_maybe=internal_machine, env=env)))
        draw_graph(file_name="3",
                   graph=internal_machine.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))
        params = HAMParamsCommon(env)
        ham_runner(ham=machine, num_episodes=global_env.episodes_count, env=env, params=params)
        rewards = params.logs["ep_rewards"]
    else:
        raise KeyError

    if rewards is not None:
        full_name = "_" + global_env.__class__.__name__
        with open(full_name + " cumulative_reward.txt", "w") as w:
            for out in get_cumulative_rewards(rewards=rewards):
                w.write(str(out) + '\n', )

        with open(full_name + " reward.txt", "w") as w:
            for out in rewards:
                w.write(str(out) + '\n', )


def main():
    for global_env in EnvironmentsArticle().environments:
        run(global_env)


def add_action_transitions(transitions):
    res = MachineGraph(transitions=transitions)
    stop = res.get_stop()
    for vertex in res.get_special_vertices(Action):
        if not res.vertex_mapping[vertex] and not res.vertex_reverse_mapping[vertex]:
            continue
        if 1 not in res.action_vertex_label_mapping[vertex]:
            res.transitions.append(MachineRelation(left=vertex, right=stop, label=1))
    return res.transitions


class M1(AbstractMachine):
    def __init__(self, env: MazeWorldEpisodeLength):
        stop = Stop()
        up1 = Action(env.ACTIONS.UP)
        up2 = Action(env.ACTIONS.UP)
        transitions = [
            MachineRelation(left=Start(), right=up1),
            MachineRelation(left=up1, right=up2, label=0),
            MachineRelation(left=up2, right=stop, label=0),
        ]
        transitions = add_action_transitions(transitions)

        super().__init__(graph=MachineGraph(transitions=transitions))


class M2(AbstractMachine):
    def __init__(self, env: MazeWorldEpisodeLength):
        stop = Stop()
        left = Action(env.ACTIONS.LEFT)
        up = Action(env.ACTIONS.UP)
        choice = Choice()

        transitions = [
            MachineRelation(left=Start(), right=left),
            MachineRelation(left=left, right=choice, label=0),
            MachineRelation(left=choice, right=left),
            MachineRelation(left=choice, right=up),
            MachineRelation(left=up, right=stop, label=0),
        ]
        transitions = add_action_transitions(transitions)

        super().__init__(graph=MachineGraph(transitions=transitions))


class M3(AbstractMachine):
    def __init__(self, env: MazeWorldEpisodeLength):
        stop = Stop()
        left1 = Action(env.ACTIONS.LEFT)
        up2 = Action(env.ACTIONS.UP)
        transitions = [
            MachineRelation(left=Start(), right=left1),
            MachineRelation(left=left1, right=up2, label=0),
            MachineRelation(left=up2, right=stop, label=0),
        ]
        transitions = add_action_transitions(transitions)

        super().__init__(graph=MachineGraph(transitions=transitions))


if __name__ == '__main__':
    main()
