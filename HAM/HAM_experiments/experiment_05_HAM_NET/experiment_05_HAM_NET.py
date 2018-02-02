import random
from collections import namedtuple, defaultdict

from gym import spaces

from HAM.HAM_core import RandomMachine, MachineGraph, Start, Stop, Action, AutoBasicMachine, MachineRelation, Choice, Call, AbstractMachine, LoopInvokerMachine, \
    RootMachine
from HAM.HAM_experiments.HAM_utils import HAMParamsCommon, ham_runner, plot_multi, PlotParams
from HAM.HAM_experiments.experiment_04_auto_random_HAM_on_maze_env.experiment_04 import is_it_machine_runnable
from environments.arm_env.arm_env import ArmEnv
from environments.env_core import CoreEnv
from environments.env_utils import EnvForTesting
from environments.weak_methods import q_learning
from utils.graph_drawer import draw_graph

# maze = generate_maze_please(size_x=2, size_y=2)
# env = MazeWorldEpisodeLength(maze=maze,finish_reward=1000)
from utils.plotting import plot_multi_test


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


class StupidMachine(AbstractMachine):
    def __init__(self):
        action = Action(action=0)
        transition = (
            MachineRelation(left=Start(), right=action),
            MachineRelation(left=action, right=action, label=0),
            MachineRelation(left=action, right=Stop(), label=1),

        )
        super().__init__(graph=MachineGraph(transitions=transition))


class HAMsNet(CoreEnv):
    ACTIONS = namedtuple("ACTIONS",
                         ["SEED_PLUS_0",
                          "SEED_PLUS_1",
                          "SEED_PLUS_2",
                          "SEED_PLUS_3",
                          "SEED_PLUS_4",
                          "DELETE_TRANSITION_TO_STOP"])(
        SEED_PLUS_0=0,
        SEED_PLUS_1=1,
        SEED_PLUS_2=2,
        SEED_PLUS_3=3,
        SEED_PLUS_4=4,
        DELETE_TRANSITION_TO_STOP=5,

    )

    def __init__(self, env, num_of_episodes, max_vertex_to_add, max_edges_to_add, init_seed=0):
        self.init_seed = init_seed
        self._reset()
        self.env = env
        self.num_of_episodes = num_of_episodes

        self.max_vertex_to_add = max_vertex_to_add
        self.max_edges_to_add = max_edges_to_add

        self.dp = {}

    def _reset(self):
        self.seed = self.init_seed
        self.machine = RandomMachine()
        self.state = tuple()
        self.last_reward = 0
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        # TODO implement done
        self._done = False

        self.vertex_added = 0
        self.edges_added = 0

    def change_graph(self):
        if self.vertex_added < self.max_vertex_to_add:
            self.machine = self.machine.with_new_vertex(env=self.env)
            self.vertex_added += 1
        else:
            try:
                self.machine = self.machine.with_new_relation()
            # TODO rewrite catching assertion to ErorObject
            except AssertionError:
                pass
            self.edges_added += 1

    def _step(self, action):
        self.state = self.state + tuple([action])
        old_seed = random.seed
        if action == self.ACTIONS.SEED_PLUS_1:
            self.seed += 1
            random.seed(self.seed)

            # CODE -------------------------------
            self.change_graph()

        elif action == self.ACTIONS.SEED_PLUS_2:
            self.seed += 2
            random.seed(self.seed)

            # CODE -------------------------------
            self.change_graph()
        elif action == self.ACTIONS.SEED_PLUS_3:
            self.seed += 3
            random.seed(self.seed)

            # CODE -------------------------------
            self.change_graph()

        elif action == self.ACTIONS.SEED_PLUS_4:
            self.seed += 4
            random.seed(self.seed)

            # CODE -------------------------------
            self.change_graph()
        elif action == self.ACTIONS.SEED_PLUS_0:
            if self.vertex_added < self.max_vertex_to_add:
                self.vertex_added += 1
            else:
                self.edges_added += 1
        elif action == self.ACTIONS.DELETE_TRANSITION_TO_STOP:
            if self.vertex_added < self.max_vertex_to_add:
                self.vertex_added += 1
            else:
                self.edges_added += 1
            new_transitions = []
            for transition in self.machine.graph.transitions:
                if not isinstance(transition.right, Stop):
                    new_transitions.append(transition)
            self.machine = RandomMachine(graph=MachineGraph(transitions=new_transitions, vertices=self.machine.graph.vertices))
        else:
            raise KeyError

        random.seed(old_seed)
        self.ham = RootMachine(LoopInvokerMachine(machine_to_invoke=super_runner(self.machine, self.env)))
        # self.ham = RootMachine(LoopInvokerMachine(machine_to_invoke=self.machine))

        reward = None
        if is_it_machine_runnable(self.machine):

            params = HAMParamsCommon(self.env)
            try:
                if self.state not in self.dp:
                    ham_runner(ham=self.ham,
                               num_episodes=self.num_of_episodes,
                               env=self.env, params=params,
                               no_output=True
                               )
                    reward = sum(params.logs["ep_rewards"])
                    if len(self.machine.graph.transitions) > 3:
                        draw_graph("pics/" + str(reward).rjust(10, "0") + str(self.state) + " " + str(self.init_seed),
                               self.machine.get_graph_to_draw(action_to_name_mapping=self.env.get_actions_as_dict()))
                    self.dp[self.state] = reward
                else:
                    reward = self.dp[self.state]
            except KeyError:
                pass
                # print("keyError", end="")
            # except AssertionError:
            #     pass
                # print("assertion", end="")
            except BlockingIOError:
                pass
        observation = self.state

        if reward is not None:
            self.last_reward = reward
        else:
            if None not in self.dp:
                params = HAMParamsCommon(self.env)
                ham_runner(ham=RootMachine(LoopInvokerMachine(machine_to_invoke=super_runner(StupidMachine(), self.env))),
                           num_episodes=self.num_of_episodes,
                           env=self.env, params=params,
                           no_output=True
                           )
                reward = sum(params.logs["ep_rewards"])
                self.dp[None] = reward
            else:
                reward = self.dp[None]
        if self.vertex_added == self.max_vertex_to_add:
            print(self.state, reward)
        info = None
        assert (self.vertex_added <= self.max_vertex_to_add)
        assert (self.edges_added <= self.max_edges_to_add)
        if self.vertex_added == self.max_vertex_to_add and self.edges_added == self.max_edges_to_add:
            self._done = True
        return observation, reward, self._done, info

    def get_current_state(self):
        return self.state

    def is_done(self):
        return self._done

    def get_actions_as_dict(self):
        return {_: getattr(self.ACTIONS, _) for _ in self.ACTIONS._fields}

    def _render(self, mode='human', close=False):
        pass


def graph_me(steps):
    env = ArmEnv(size_x=6, size_y=3, cubes_cnt=3, episode_max_length=300, finish_reward=200, action_minus_reward=-1, tower_target_size=2)
    net = HAMsNet(env=env, num_of_episodes=300, max_vertex_to_add=7, max_edges_to_add=6)
    for i in steps:
        net.step(i)
    draw_graph("ololo", net.machine.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))


def main():
    env_obj = EnvForTesting()
    net = HAMsNet(env=env_obj.env, num_of_episodes=env_obj.episodes, max_vertex_to_add=8, max_edges_to_add=7, init_seed=random.randrange(1021321321))

    q_table = defaultdict(lambda: 0)
    to_plot = []
    for i in range(10):
        q_stats, q_table = q_learning(env=net, num_episodes=200, gamma=1, eps=1 - i * 10 / 10, q_table=q_table, alpha=0.5)
        to_plot += q_stats
        print(":::::" * 10)
        print(1 - i * 10 / 100)

    plot_multi_test([to_plot])


if __name__ == '__main__':
    # graph_me(steps=(1, 0, 1, 3, 0, 3, 1, 1, 2, 1, 0, 2))
    main()
