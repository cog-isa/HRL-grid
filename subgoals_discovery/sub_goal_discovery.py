from collections import defaultdict

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

from HAM.HAM_utils import HAMParamsCommon
from cozy_RL.cozy_RL import CozyRL, Q_Agent, RewardChartDrawer, Agent
from subgoals_discovery.two_rooms_env import TwoRooms, AutoMachineSimple
import numpy as np

class SubGoalDiscovery(Agent):
    def __init__(self, env, name):
        self.machine = AutoMachineSimple(env)
        self.params = HAMParamsCommon(env)
        self.name = name

        self.bns_count = defaultdict(lambda: 0)
        self.bn_added = None

    def pre_episode(self, info):
        self.bn_added = {info['ps']:1}

    def make_action(self, info):
        action = self.machine.run(self.params)
        return action

    def post_action(self, info):
        cs = info["cs"]
        self.bn_added[cs] = 1
        self.machine.update_after_action(info["r"])

    def post_episode(self, info):
        # TODO run if it was only successful episode
        for state in self.bn_added:
            if self.bn_added[state] is not None:
                self.bns_count[state] += 1

        if (info["episode"] + 1) % 1000 == 0:
            bns_count = self.bns_count
            self.bns_count = defaultdict(lambda: 0)
            V = self.machine.V
            env = info["env"]

            def get_clusters(V, n_clusters, affinity):
                states = sorted(V.keys())
                ss = {"state": states}
                # noinspection PyTypeChecker
                for i in range(len(V[states[0]])):
                    ss[str(i)] = [V[_][i] for _ in states]
                df = pd.DataFrame(ss).set_index("state")
                sc = MinMaxScaler()
                df = df.rename(index=str, columns={"0": "x", "1": "y", "2": 'V'})
                X = df[["x", "y", "V"]]
                # X[["V"]] *= 0.5
                # df[["x", "y"]] = df[["x", "y"]].apply(np.float)
                df["x"] = df["x"].astype(np.float)
                df["y"] = df["y"].astype(np.float)

                sc.fit(pd.np.vstack((df[["x"]], df[["y"]])))

                df[["x", "y"]] = sc.transform(df[["x", "y"]])
                ag = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity)
                clustered = list(ag.fit_predict(X))
                cluster_state_mapping = {}
                for i in range(len(states)):
                    cluster_state_mapping[states[i]] = clustered[i]
                return cluster_state_mapping

            # all_states = V.keys()
            n_clusters = 4
            map_state_to_cluster = get_clusters(V=V, n_clusters=n_clusters, affinity="euclidean")

            def get_bns_in_increasing_order(bns_count):
                state_count_pairs = sorted([(bns_count[_], _) for _ in bns_count], reverse=True)
                return list(map(lambda x: x[1], state_count_pairs, ))
            map_state_to_cluster[518] = 0
            print("x, y:", env.decode(518))
            def get_mapping_for_cluster_to_sorted_bns(sorted_bns, map_state_to_cluster):
                res = defaultdict(lambda: list())
                for state in sorted_bns:
                    res[map_state_to_cluster[state]].append(state)
                return res

            # bns = bottlenecks
            sorted_bns = get_bns_in_increasing_order(bns_count=bns_count)
            map_cluster_to_sorted_bns = get_mapping_for_cluster_to_sorted_bns(sorted_bns=sorted_bns,
                                                                              map_state_to_cluster=map_state_to_cluster)

            env.mark = {}

            for current_state in map_state_to_cluster:
                env.mark[current_state] = str(map_state_to_cluster[current_state])

            class colors:
                HEADER = '\033[95m'
                OKBLUE = '\033[94m'
                OKGREEN = '\033[92m'
                WARNING = '\033[93m'
                FAIL = '\033[91m'
                ENDC = '\033[0m'
                BOLD = '\033[1m'
                UNDERLINE = '\033[4m'

                COLOR_LIST = [HEADER, OKBLUE, OKGREEN, WARNING, FAIL]

            # draw best bns for clusters
            BNS_FOR_CLUSTER = 5
            for q in map_cluster_to_sorted_bns:
                for j in map_cluster_to_sorted_bns[q][:BNS_FOR_CLUSTER]:
                    env.mark[j] = colors.COLOR_LIST[q % len(colors.COLOR_LIST)] + str(q) + colors.ENDC
            env.render()
            env.mark = {}


def main():
    env = TwoRooms()
    cozy = CozyRL(environment=env, agents=[
        SubGoalDiscovery(env=env, name="subgoal discovery"),
        Q_Agent(env, name="Q-1", eps=0.5),
        # Q_Agent(env=TwoRooms(), name="Q-2", eps=0.6),
        # Q_Agent(env=TwoRooms(), name="Q-3", eps=0.7),
        # StandardHAM(env=env, name="stHAM"),

    ],
                  supplementary=[RewardChartDrawer(smooth_step=10)],
                  number_of_episodes=2000)
    cozy.run()


if __name__ == '__main__':
    main()
