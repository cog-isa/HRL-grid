import random
from collections import namedtuple, defaultdict

import itertools

import numpy as np
import sys

from HAM.HAM_core import RandomMachine, MachineGraph, Start, Stop, Action, AutoBasicMachine, MachineRelation, Choice, Call, AbstractMachine, LoopInvokerMachine, \
    RootMachine
from HAM.HAM_experiments.HAM_utils import HAMParamsCommon, ham_runner, plot_multi, PlotParams
from HAM.HAM_experiments.experiment_04_auto_random_HAM_on_maze_env.experiment_04 import is_it_machine_runnable
from environments.arm_env.arm_env import ArmEnv
from environments.env_core import CoreEnv
from environments.grid_maze_env.grid_maze_generator import generate_maze_please
from environments.grid_maze_env.maze_world_env import MazeWorldEpisodeLength
from utils.graph_drawer import draw_graph


# maze = generate_maze_please(size_x=2, size_y=2)
# env = MazeWorldEpisodeLength(maze=maze,finish_reward=1000)

def arg_max_action(q_dict, state, action_space):
    result_action = 0
    for action_to in range(action_space):
        if q_dict[state, action_to] > q_dict[state, result_action]:
            result_action = action_to
    return result_action


def q_learning(env, num_episodes, eps=0.1, alpha=0.1, gamma=0.9):
    to_plot = []

    # initialize q-function
    q_table = defaultdict(lambda: 0)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        ep_reward = 0
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
            eps = eps - 0.01 * eps

        # Reset the environment and pick the first state
        state = env.reset()

        for t in itertools.count():
            # E-greedy
            if np.random.rand(1) < eps:
                # choosing a random action
                action = np.random.choice(env.action_space.n, size=1)[0]
            else:
                # choosing arg_max action
                action = arg_max_action(q_dict=q_table, state=state, action_space=env.action_space.n)

            next_state, reward, done, _ = env.step(action)
            # print(q_table[state, action])
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (
                reward + gamma * q_table[next_state, arg_max_action(q_dict=q_table, state=next_state, action_space=env.action_space.n)])

            # Update statistics
            ep_reward += reward

            if done:
                break

            state = next_state
        to_plot.append(ep_reward)
    return to_plot, q_table


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


class HAMsNet(CoreEnv):
    ACTIONS = namedtuple("ACTIONS", ["SEED_PLUS_1_ADD_VERTEX", "SEED_PLUS_2_ADD_VERTEX", "SEED_PLUS_3_ADD_EDGE", "SEED_PLUS_4_ADD_EDGE"])(
        SEED_PLUS_1_ADD_VERTEX=0,
        SEED_PLUS_2_ADD_VERTEX=1,
        SEED_PLUS_3_ADD_EDGE=2,
        SEED_PLUS_4_ADD_EDGE=3
    )

    def __init__(self, env):
        self._reset()
        self.env = env

    def _reset(self):
        self.seed = 0
        self.machine = RandomMachine()
        self.state = tuple()
        self.last_reward = 0
        # TODO implement done
        self._done = False

    def _step(self, action):
        self.state = self.state + tuple([action])
        old_seed = random.seed
        if action == self.ACTIONS.SEED_PLUS_1_ADD_VERTEX:
            self.seed += 1
            random.seed(self.seed)
            try:
                self.machine = self.machine.with_new_relation()
            # TODO rewrite catching assertion to ErorObject
            except AssertionError:
                pass

        elif action == self.ACTIONS.SEED_PLUS_2_ADD_VERTEX:
            self.seed += 2
            random.seed(self.seed)
            try:
                self.machine = self.machine.with_new_relation()
            # TODO rewrite catching assertion to ErorObject
            except AssertionError:
                pass
        elif action == self.ACTIONS.SEED_PLUS_3_ADD_EDGE:
            self.seed += 3
            random.seed(self.seed)
            self.machine = self.machine.with_new_vertex(env=self.env)
        elif action == self.ACTIONS.SEED_PLUS_4_ADD_EDGE:
            self.seed += 4
            random.seed(self.seed)
            self.machine = self.machine.with_new_vertex(env=self.env)
        else:
            raise KeyError
        random.seed(old_seed)
        self.ham = RootMachine(LoopInvokerMachine(machine_to_invoke=super_runner(self.machine, self.env)))
        num_episodes = 1000
        reward = None
        if is_it_machine_runnable(self.machine):
            params = HAMParamsCommon(self.env)
            try:
                ham_runner(ham=self.ham,
                           num_episodes=num_episodes,
                           env=self.env, params=params, no_output=True)
                reward = sum(params.logs["ep_rewards"])
            except KeyError:
                pass
                # print("keyError", end="")
            except AssertionError:
                pass
                # print("assertion", end="")
        observation = self.state
        if reward is not None:
            return_reward = reward
            self.last_reward = reward
        else:
            return_reward = self.last_reward
        info = None
        if len(self.state) > 10:
            self._done = True
        return observation, return_reward, self._done, info

    def get_current_state(self):
        return self.state

    def is_done(self):
        return self._done

    def get_actions_as_dict(self):
        return {_: getattr(self.ACTIONS, _) for _ in self.ACTIONS._fields}

    def _render(self, mode='human', close=False):
        pass


def main():
    env = ArmEnv(size_x=3, size_y=3, cubes_cnt=3, episode_max_length=600, finish_reward=200, action_minus_reward=-1, tower_target_size=2)
    net = HAMsNet(env=env)
    p = HAMParamsCommon(net)
    ham_runner(ham=AutoBasicMachine(env=net), num_episodes=3000, env=net, params=p)
    draw_graph("123", net.machine.get_graph_to_draw())
    to_plot = []
    to_plot.append(PlotParams(curve_to_draw=p.logs["ep_rewards"], label="HAM_with_pull_up"))

    plot_multi(to_plot)


if __name__ == '__main__':
    main()
