from collections import defaultdict
from time import sleep

import numpy as np
import pandas as pd
from gym.core import Env
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import seaborn as sns

from HAM.HAM_utils import HAMParamsCommon
from subgoals_discovery.two_rooms_env import TwoRooms, AutoMachineSimple


class CozyRL:
    def __init__(self, environment: Env, agents, supplementary, number_of_episodes):
        self.environment = environment
        self.agents = agents
        self.supplementary = supplementary
        self.number_of_episodes = number_of_episodes

    def run(self):
        info = {}
        # pre learning
        for listener in self.agents + self.supplementary:
            listener.pre_learning(info)
        for agent in self.agents:
            desc = "{agent.name}".format(**locals())
            bar_format = '{l_bar}[{bar}]'
            info['agent'] = agent
            # pre agent learning
            for listener in self.supplementary + [agent]:
                listener.pre_agent_learning(info)

            for episode in tqdm(range(self.number_of_episodes), bar_format=bar_format, desc=desc, position=0):
            # for episode in range(self.number_of_episodes):
                info['episode'] = episode

                cs = None
                ps = self.environment.reset()
                info["cs"] = info["ps"] = ps
                # pre episode
                for listener in self.supplementary + [agent]:
                    listener.pre_episode(info)

                done = False
                while not done:
                    info["cs"] = ps
                    info["env"] = self.environment
                    info["ps"] = ps
                    # pre action
                    for listener in self.supplementary + [agent]:
                        listener.pre_action(info)

                    action = agent.make_action(info)

                    cs, reward, done, listener = self.environment.step(action)
                    info.update({"ps": ps, "cs": cs, "r": reward, "a": action, "env": self.environment})
                    ps = cs
                    # post action
                    for listener in self.supplementary + [agent]:
                        listener.post_action(info)
                info["cs"] = cs
                # post episode
                for listener in self.supplementary + [agent]:
                    listener.post_episode(info)
            # post agent learning
            for listener in self.agents + self.supplementary:
                listener.post_agent_learning(info)

        sleep(0.1)
        # post learning
        for listener in self.agents + self.supplementary:
            listener.post_learning(info)


class CozyListener:
    def pre_learning(self, info):
        pass

    def pre_agent_learning(self, info):
        pass

    def pre_episode(self, info):
        pass

    def pre_action(self, info):
        pass

    def post_action(self, info):
        pass

    def post_episode(self, info):
        pass

    def post_agent_learning(self, info):
        pass

    def post_learning(self, info):
        pass


class StatisticsListener(CozyListener):
    def __init__(self):
        self.episode_r = None
        self.reward_for_each_episode = None
        self.current_agent = None
        # self.agent_name_to_reward = {}
        self.agents_statistics = []

    def pre_agent_learning(self, info):
        self.current_agent = info["agent"]
        self.reward_for_each_episode = []

    def pre_episode(self, info):
        self.episode_r = 0

    def post_action(self, info):
        self.episode_r += info["r"]

    def post_episode(self, info):
        self.reward_for_each_episode.append(self.episode_r)

    def post_agent_learning(self, info):
        class AgentAndRewards:
            def __init__(self, agent, rewards):
                self.agent = agent
                self.name = agent.name
                self.rewards = rewards

        self.agents_statistics.append(AgentAndRewards(agent=self.current_agent, rewards=self.reward_for_each_episode))


class RewardChartDrawer(StatisticsListener):

    def __init__(self, smooth_step=10):
        super().__init__()
        self.smooth_step = smooth_step

    def post_learning(self, info):

        sns.set(palette="Set2")

        plt.title("Statistics")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        # plt.ylim(bottom=-20)
        for agent in self.agents_statistics:
            name = agent.name
            x = agent.rewards

            means = []
            upper = []
            lower = []

            for i in range(0, len(x), self.smooth_step):
                t = []
                for j in range(i, min(len(x), i + self.smooth_step)):
                    t.append(x[j])
                t = np.array(t)
                means.append(np.mean(t))
                upper.append(np.abs((np.mean(t) - np.max(t))))
                lower.append(np.abs((np.mean(t) - np.min(t))))

            means = np.array(means)

            # color = sns.color_palette()[2]
            plt.plot(np.array(range(len(means))) * self.smooth_step, means, label=str(name))

            plt.fill_between(np.array(range(len(means))) * self.smooth_step, means - lower, means + upper, alpha=0.15)
            plt.legend()
        plt.show()


class Agent(CozyListener):
    def make_action(self, info):
        raise NotImplementedError


class Q_Agent(Agent):
    def __init__(self, env, name, eps=0.1):
        self.q_table = defaultdict(lambda: 0)
        self.alpha = 0.1
        self.gamma = 0.9
        self.eps = eps
        self.name = name

    def pre_episode(self, info):
        self.eps = self.eps * 0.999
        # self.alpha = self.alpha * 0.99999

    def make_action(self, info):

        cs = info["cs"]
        env = info["env"]
        if np.random.rand(1) < self.eps:
            action = np.random.choice(env.action_space.n, size=1)[0]
        else:
            action = self.arg_max_action(q_dict=self.q_table, state=cs, action_space=env.action_space.n)
        return action

    def post_action(self, info):
        ps = info["ps"]
        cs = info["cs"]
        r = info["r"]
        pa = info["a"]
        env = info["env"]

        ca = self.arg_max_action(q_dict=self.q_table, state=cs, action_space=env.action_space.n)
        self.q_table[ps, pa] = (1 - self.alpha) * self.q_table[ps, pa] + self.alpha * (
                r + self.gamma * self.q_table[cs, ca])

    @staticmethod
    def arg_max_action(q_dict, state, action_space):
        result_action = 0
        for action_to in range(action_space):
            if q_dict[state, action_to] > q_dict[state, result_action]:
                result_action = action_to
        return result_action


class StandardHAM(Agent):
    def __init__(self, env, name):
        self.machine = AutoMachineSimple(env)
        self.params = HAMParamsCommon(env)
        self.name = name

    def make_action(self, info):
        action = self.machine.run(self.params)
        return action

    def post_action(self, info):
        # if info["r"] > 0.5:
        #     print(info["r"])
        self.machine.update_after_action(info["r"])




