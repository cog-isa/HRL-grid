import random
import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

import pandas as pd

primitive_actions = LEFT, RIGHT, UP, DOWN, PICK_UP, DROP_OFF = list(range(6))
composite_actions = GO_TO_SOURCE, PUT, GET, ROOT, GO_TO_DESTINATION = list(range(6, 11))


class Agent:

    def __init__(self, env, alpha, gamma, eps):
        """
        Agent initialisation
        :param env: used environment (taxi domain)
        :param alpha: learning rate
        :param gamma: discount rate
        :param eps: rate for e-greedy policy
        """
        self.env = env
        action_size = len(primitive_actions) + len(composite_actions)
        self.V = np.zeros((action_size, env.observation_space.n))
        self.C = np.zeros((action_size, env.observation_space.n, action_size))

        self.graph = [[] for _ in range(len(primitive_actions) + len(composite_actions))]

        self.graph[GO_TO_SOURCE] = self.graph[GO_TO_DESTINATION] = [LEFT, RIGHT, UP, DOWN]
        self.graph[PUT] = [DROP_OFF, GO_TO_DESTINATION]
        self.graph[GET] = [PICK_UP, GO_TO_SOURCE]
        self.graph[ROOT] = [PUT, GET]

        self.alpha = alpha
        self.gamma = gamma
        self.r_sum = 0
        self.new_s = self.env.s
        self.done = False
        self.eps = eps

    @staticmethod
    def in_car(pass_index):
        """
        checking what the  passenger in car
        :param pass_index: pass_index from env.decode(self.env.s)
        :return: boolean
        """
        return pass_index == 4

    def is_terminal(self, node):
        """
        checking current node of tree for termination
        :param node: current node
        :return:
        """
        taxi_row, taxi_col, pass_idx, destination = list(self.env.decode(self.env.s))
        if node == ROOT:
            return self.done
        elif node == GET:
            return self.in_car(pass_idx)
        elif node == PUT:
            return not self.in_car(pass_idx) or self.done
        elif node == GO_TO_SOURCE:
            return self.env.locs[pass_idx] == (taxi_row, taxi_col)
        elif node == GO_TO_DESTINATION:
            return self.env.locs[destination] == (taxi_row, taxi_col)

    def evaluate(self, node, s):
        """
        evaluating best node for transition, implementation of evaluate function from Dietrich's paper
        :param node: current node
        :param s: state of the environment
        :return: best value for step from current node and index of that edge
        """
        if node in primitive_actions:
            return self.V[node, s], node
        elif node in composite_actions:
            # TODO to reassign variables with more clear names
            j_arg_max, cur_max = None, None
            for j, a in enumerate(self.graph[node]):
                v, _ = self.evaluate(a, s)
                if cur_max is None or v + self.C[node, s, a] > cur_max:
                    j_arg_max = j
                    cur_max = v + self.C[node, s, a]
            return cur_max, j_arg_max
        else:
            raise KeyError

    def greed_act(self, node, s):
        """
        choosing greedy transition on tree
        :param node:  current node
        :param s: current environment state
        :return: action index
        """
        # TODO rewrite this code
        q = np.arange(0)
        for a2 in self.graph[node]:
            q = np.concatenate((q, [self.V[a2, s] + self.C[node, s, a2]]))
        max_arg = np.argmax(q)
        possible_a = np.array(list(self.graph[node]))
        if np.random.rand(1) < self.eps:
            return np.random.choice(possible_a)
        else:
            return possible_a[max_arg]

    def is_parent_terminates(self, node):
        """
        checking for parents termination does not  implemented for max_q_0 implementation for Taxi domain
        since its redundantly. So now it always returns False
        :param node: current node of tree
        :return: boolean value
        """
        return False

    def max_q_0(self, node, s):
        """
        max_q_0 algorithm
        :param node: current node of tree
        :param s: current state of environment
        :return: number of applied primitive actions
        """
        self.done = False
        if node in primitive_actions:
            _, r, self.done, _ = self.env.step(node)
            self.r_sum += r
            self.V[node, s] += self.alpha * (r - self.V[node, s])
            self.eps *= 0.999
            return 1
        elif node in composite_actions:

            count = 0
            while not self.is_terminal(node) and not self.is_parent_terminates(node):
                a = self.greed_act(node, s)
                for _ in range(100):
                    if not self.is_terminal(a):
                        break
                    a = random.choice(list(self.graph[node]))
                else:
                    raise ValueError("can't choose next vertex which doesn't terminates on current state")
                self.alpha *= 0.99999
                n = self.max_q_0(a, s)
                obs = self.env.s
                v, _ = self.evaluate(node, obs)
                self.C[node, s, a] += self.alpha * (self.gamma ** n * v - self.C[node, s, a])

                count += n
                s = obs
            return count
        else:
            raise KeyError

    def reset(self):
        """
        resetting current environment and special variables
        :return: None
        """
        self.env.reset()
        self.r_sum = 0
        self.done = False


def run_max_q(episodes):
    """
    launches max_q
    :param episodes: number of episodes
    :return: list of rewards for episodes
    """
    env = gym.make('Taxi-v2').env
    taxi = Agent(env=env, alpha=0.8, gamma=0.99, eps=0.2)
    rewards_for_episodes = []
    for _ in tqdm(range(episodes), postfix="MAX_Q_0"):
        taxi.reset()
        taxi.max_q_0(ROOT, env.s)
        rewards_for_episodes.append(taxi.r_sum)
    return rewards_for_episodes


def run_q_learning(episodes):
    """
    launches q-learning algorithm
    :param episodes: number of episodes
    :return: list of rewards for episodes
    """
    from environments.weak_methods import q_learning
    env = gym.make('Taxi-v2').env
    to_plot, _ = q_learning(env=env, num_episodes=episodes, eps=0.1, alpha=0.1, gamma=0.9)
    return to_plot


def main():
    """
    :return: None
    """
    sns.set(palette="Set2")
    episodes = 100
    tests = 5
    stack = np.dstack([[p(episodes) for _ in range(tests)] for p in [run_max_q, run_q_learning]])
    name = pd.Series(["MAX-Q", "Q-learning"], name="")
    sns.tsplot(stack, condition=name, value="position")

    # If you want to save:
    # plt.savefig("MAXQ_vs_Q.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
