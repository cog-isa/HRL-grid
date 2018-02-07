import random
from collections import namedtuple, defaultdict

from gym import spaces

from HAM.HAM_core import RandomMachine, MachineGraph, Start, Stop, Action, AutoBasicMachine, MachineRelation, Choice, Call, AbstractMachine, LoopInvokerMachine, \
    RootMachine
from HAM.HAM_experiments.HAM_utils import HAMParamsCommon, ham_runner, plot_multi, PlotParams
from HAM.HAM_experiments.experiment_04_auto_random_HAM_on_maze_env.experiment_04 import is_it_machine_runnable
from HAM.HAM_experiments.experiment_05_HAM_NET.experiment_05_HAM_NET import super_runner
from environments.arm_env.arm_env import ArmEnv
from environments.env_core import CoreEnv
from environments.env_utils import EnvForTesting, EnvForTesting2
from environments.weak_methods import q_learning
from utils.graph_drawer import draw_graph

# maze = generate_maze_please(size_x=2, size_y=2)
# env = MazeWorldEpisodeLength(maze=maze,finish_reward=1000)
from utils.plotting import plot_multi_test


# def super_runner(call_me_maybe, env):
#     start = Start()
#     choice_one = Choice()
#     actions = [Action(action=_) for _ in env.get_actions_as_dict().values()]
#     stop = Stop()
#
#     call = Call(call_me_maybe)
#     transitions = [MachineRelation(left=start, right=choice_one), ]
#     for action in actions:
#         transitions.append(MachineRelation(left=choice_one, right=action))
#         transitions.append(MachineRelation(left=action, right=stop, label=0))
#         transitions.append(MachineRelation(left=action, right=stop, label=1))
#     transitions.append(MachineRelation(left=choice_one, right=call))
#     transitions.append(MachineRelation(left=call, right=stop))
#
#     return AbstractMachine(graph=MachineGraph(transitions=transitions))


class StupidMachine(AbstractMachine):
    def __init__(self):
        action = Action(action=0)
        transition = (
            MachineRelation(left=Start(), right=action),
            MachineRelation(left=action, right=action, label=0),
            MachineRelation(left=action, right=Stop(), label=1),

        )
        super().__init__(graph=MachineGraph(transitions=transition))


class HAMsNet2(CoreEnv):
    ACTIONS = namedtuple("ACTIONS",
                         ["ACTION_01",
                          "ACTION_02",
                          "ACTION_03",
                          "ACTION_04",
                          "ACTION_05",
                          "ACTION_06"])(
        ACTION_01=0,
        ACTION_02=1,
        ACTION_03=2,
        ACTION_04=3,
        ACTION_05=4,
        ACTION_06=5,
    )

    def __init__(self, env, num_of_episodes, max_size):
        self.machine = None
        self._reset()
        self.env = env
        self.num_of_episodes = num_of_episodes
        self.max_size = max_size
        self.dp = {}

    def _reset(self):
        self.machine = RandomMachine()
        self.state = tuple()
        self.last_reward = 0
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        # TODO implement done
        self._done = False

        self.vertex_added = 0
        self.edges_added = 0

        self.machine = RandomMachine(graph=MachineGraph(transitions=[MachineRelation(left=Start(), right=Stop())]))

    def add(self, action):
        transitions = []

        for relation in self.machine.graph.transitions:
            if isinstance(relation.right, Stop) and (relation.label == 0 or isinstance(relation.left, Start)):
                a = Action(action=action)
                if relation.label == 0:
                    transitions.append(MachineRelation(left=relation.left, right=a, label=0))
                else:
                    transitions.append(MachineRelation(left=relation.left, right=a))
                transitions.append(MachineRelation(left=a, right=self.machine.graph.get_stop(), label=0))
            else:
                transitions.append(relation)
        res = MachineGraph(transitions=transitions)

        for vertex in res.get_special_vertices(Action):
            # print("::", res.graph.action_vertex_label_mapping[vertex])
            if not res.vertex_mapping[vertex] and not res.vertex_reverse_mapping[vertex]:
                continue
            if 1 not in res.action_vertex_label_mapping[vertex]:
                res.transitions.append(MachineRelation(left=vertex, right=res.get_stop(), label=1))

        self.machine = RandomMachine(graph=MachineGraph(transitions=transitions))

    def _step(self, action):
        self.state = self.state + tuple([action])

        self.ham = RootMachine(LoopInvokerMachine(machine_to_invoke=super_runner(self.machine, self.env)))
        reward = None

        if action is None:
            raise KeyError
        elif action == self.ACTIONS.ACTION_01:
            self.add(Action(action=action))
        elif action == self.ACTIONS.ACTION_02:
            self.add(Action(action=action))
        elif action == self.ACTIONS.ACTION_03:
            self.add(Action(action=action))
        elif action == self.ACTIONS.ACTION_04:
            self.add(Action(action=action))
        elif action == self.ACTIONS.ACTION_05:
            self.add(Action(action=action))
        elif action == self.ACTIONS.ACTION_06:
            self.add(Action(action=action))

        if is_it_machine_runnable(self.machine):
            if self.state in self.dp:
                reward = self.dp[self.state]
            else:
                params = HAMParamsCommon(self.env)
                ham_runner(ham=self.ham,
                           num_episodes=self.num_of_episodes,
                           env=self.env, params=params,
                           no_output=True
                           )
                reward = sum(params.logs["ep_rewards"])
                self.dp[self.state] = reward
            draw_graph("pics/" + str(reward).rjust(10, "0") + str(self.state) + " ",
                       self.machine.get_graph_to_draw(action_to_name_mapping=self.env.get_actions_as_dict()))

        observation = self.state
        if len(self.state) >= self.max_size:
            self._done = True

        return observation, reward, self._done, None

    def _render(self, mode='human', close=False):
        pass


def get_current_state(self):
    return self.state


def is_done(self):
    return self._done


def get_actions_as_dict(self):
    return {_: getattr(self.ACTIONS, _) for _ in self.ACTIONS._fields}


def main():
    global_env = EnvForTesting2()
    env_obj = global_env.env
    net = HAMsNet2(env=env_obj, num_of_episodes=global_env.episodes, max_size=2)
    q_table = None
    q_stats, q_table = q_learning(env=net, num_episodes=500, gamma=1, eps=0.9, alpha=0.5, q_table=q_table)
    net.max_size = 3
    q_stats, q_table = q_learning(env=net, num_episodes=500, gamma=1, eps=0.5, alpha=0.5, q_table=q_table)
    net.max_size = 4
    q_stats, q_table = q_learning(env=net, num_episodes=500, gamma=1, eps=0.3, alpha=0.5, q_table=q_table)
    net.max_size = 5
    q_stats, q_table = q_learning(env=net, num_episodes=500, gamma=1, eps=0.2, alpha=0.5, q_table=q_table)
    net.max_size = 6
    q_stats, q_table = q_learning(env=net, num_episodes=500, gamma=1, eps=0.1, alpha=0.4, q_table=q_table)
# 0035768353 (q-learning)
# 0035786236(3, 1)

if __name__ == '__main__':
    main()
