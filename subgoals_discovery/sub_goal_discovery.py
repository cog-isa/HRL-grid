from collections import defaultdict
from copy import copy

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from HAM.HAM_utils import HAMParamsCommon
from cozy_RL.cozy_RL import CozyRL, Q_Agent, RewardChartDrawer, Agent, StandardHAM
from subgoals_discovery.two_rooms_env import TwoRooms, AutoMachineSimple
import numpy as np


class SubGoalDiscovery(Agent):
    class BnsMachine:
        def __init__(self, params, cluster_index, list_of_bns, states_in_my_cluster, env):
            self.machine = AutoMachineSimple(env)
            self.cluster_index = cluster_index
            self.bns = set(list_of_bns)
            self.states_in_my_cluster = states_in_my_cluster
            self.params = params

    def __init__(self, env, name, discovery_on_episodes):
        self.machine = self.BnsMachine(params=HAMParamsCommon(env=env), cluster_index=None, list_of_bns=[],
                                       states_in_my_cluster=[], env=env)
        self.name = name

        self.bns_count = defaultdict(lambda: 0)
        self.bn_added = None

        self.hams = None
        self.current_ham = None
        self.current_ham_bn_reached = None

        self.episode_r = None

        self.discovery_on_episodes = discovery_on_episodes

        self.hams_stage_one = None
        self.hams_stage_two = None

    def pre_episode(self, info):
        self.episode_r = 0
        self.bn_added = {info['ps']: 1}
        self.current_ham = None
        if self.hams_stage_two is not None:
            for ham in self.hams_stage_two:
                ham.reward = 4
            for ham in self.hams_stage_one:
                ham.reward = 0

    def make_action(self, info):

        def select_machine_for_current_state(hams, state):
            for h in hams:
                if state in h.states_in_my_cluster:
                    return h
            raise KeyError

        cs = info["cs"]
        if self.hams_stage_one is None:
            self.machine.params.cs = cs
            return self.machine.machine.run(self.machine.params)
        else:
            if self.current_ham is None or (
                    cs not in self.current_ham.states_in_my_cluster and not self.current_ham_bn_reached):
                self.current_ham = select_machine_for_current_state(self.hams_stage_one, cs)
                self.current_ham_bn_reached = False

            if cs in self.current_ham.bns and cs in self.current_ham.states_in_my_cluster:
                self.current_ham_bn_reached = True
                self.current_ham = select_machine_for_current_state(self.hams_stage_two, cs)

            if cs not in self.current_ham.states_in_my_cluster:
                if self.current_ham_bn_reached:
                    self.current_ham_bn_reached = False
                self.current_ham = select_machine_for_current_state(self.hams_stage_one, cs)
                    # ToDO add reward

            self.current_ham.params.cs = cs, self.current_ham.reward, self.current_ham.cluster_index
            # self.current_ham.params.cs = cs, self.current_ham.cluster_index
            return self.current_ham.machine.run(self.current_ham.params)

    def post_action(self, info):
        cs = info["cs"]
        self.bn_added[cs] = 1
        self.episode_r += info["r"]
        if self.hams_stage_one is None:
            self.machine.machine.update_after_action(info["r"])
        else:
            r = info["r"]
            if self.current_ham_bn_reached:
                r += self.current_ham.reward
                self.current_ham.reward = 0
            self.current_ham.machine.update_after_action(r)

    def post_episode(self, info):
        # print(self.episode_r)
        if self.episode_r > 0:
            self.bn_added[info["cs"]] = 1
            for state in self.bn_added:
                if self.bn_added[state] is not None:
                    self.bns_count[state] += 1

        self.episode_r = 0
        # TODO add if it was only successful episode
        if info["episode"] in self.discovery_on_episodes:
            bns_count = self.bns_count
            self.bns_count = defaultdict(lambda: 0)

            if self.hams is None:
                V = self.machine.machine.V
            else:
                V = {}
                for ham in self.hams:
                    V = {**V, **ham.machine.V}
            for i in bns_count:
                if i not in V:
                    V[i] = (*info["env"].decode(i), 0)

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
                # X[["V"]] *= 0.6
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

            cluster_to_list_of_states = defaultdict(lambda: [])
            for _ in map_state_to_cluster:
                cluster_to_list_of_states[map_state_to_cluster[_]].append(_)

            def get_bns_in_decreasing_order(bns_count):
                state_count_pairs = sorted([(bns_count[_], _) for _ in bns_count], reverse=True)
                return list(map(lambda x: x[1], state_count_pairs, ))

            # map_state_to_cluster[518] = 0
            # env.render()
            # print("x, y:", env.decode(518))
            def get_mapping_for_cluster_to_sorted_bns(sorted_bns, map_state_to_cluster):
                res = defaultdict(lambda: list())
                for state in sorted_bns:
                    res[map_state_to_cluster[state]].append(state)
                return res

            # bns = bottlenecks

            def normalize_dict(di):
                res = copy(di)
                factor = 1.0 / sum(res.values())
                for k in res:
                    res[k] = res[k] * factor
                return res

            only_v = {_: V[_][2] for _ in V}
            v_for_combine = normalize_dict(only_v)
            bns_for_combine = normalize_dict(bns_count)

            combined_v_and_bns = defaultdict(lambda: min(V.values()))
            env.mark = {}
            for i in bns_count:
                combined_v_and_bns[i] = bns_for_combine[i] * 10 + v_for_combine[i] * 15
                # env.mark[i] = str(int(combined_v_and_bns[i] * 100))
            # env.render()

            sorted_bns = get_bns_in_decreasing_order(bns_count=combined_v_and_bns)
            map_cluster_to_sorted_bns = get_mapping_for_cluster_to_sorted_bns(sorted_bns=sorted_bns,
                                                                              map_state_to_cluster=map_state_to_cluster)

            env.mark = {}

            for current_state in map_state_to_cluster:
                env.mark[current_state] = str(map_state_to_cluster[current_state])

            class colors:
                HEADER = '\033[95m'
                BLUE = '\033[94m'
                GREEN = '\033[92m'
                WARNING = '\033[93m'
                FAIL = '\033[91m'
                ENDC = '\033[0m'
                BOLD = '\033[1m'
                UNDERLINE = '\033[4m'

                COLOR_LIST = [HEADER, BLUE, GREEN, WARNING, FAIL]

            # draw best bns for clusters
            BNS_FOR_CLUSTER = 1
            for q in map_cluster_to_sorted_bns:
                for j in map_cluster_to_sorted_bns[q][:BNS_FOR_CLUSTER]:
                    env.mark[j] = colors.COLOR_LIST[q % len(colors.COLOR_LIST)] + str(q) + colors.ENDC
            env.render()
            # env.mark = {}
            # hams = [self.BnsMachine(params=HAMParamsCommon(env=env),
            #                         cluster_index=_,
            #                         list_of_bns=map_cluster_to_sorted_bns[_][:BNS_FOR_CLUSTER],
            #                         states_in_my_cluster=cluster_to_list_of_states[_],
            #                         env=env
            #                         ) for _ in map_cluster_to_sorted_bns]
            params = HAMParamsCommon(env=env)
            hams_stage_one = [self.BnsMachine(params=params,
                                              cluster_index=_,
                                              list_of_bns=map_cluster_to_sorted_bns[_][:BNS_FOR_CLUSTER],
                                              states_in_my_cluster=cluster_to_list_of_states[_],
                                              env=env
                                              ) for _ in map_cluster_to_sorted_bns]
            hams_stage_two = [self.BnsMachine(params=params,
                                              cluster_index=_,
                                              list_of_bns=map_cluster_to_sorted_bns[_][:BNS_FOR_CLUSTER],
                                              states_in_my_cluster=cluster_to_list_of_states[_],
                                              env=env
                                              ) for _ in map_cluster_to_sorted_bns]

            self.hams_stage_one = hams_stage_one
            self.hams_stage_two = hams_stage_two
            # TODO REWRITE WITH REPLAY BUFFER!!!
            self.machine_for_non_clustered = self.machine

            # self.hams = hams
            self.current_ham = None


def main():
    env = TwoRooms()
    learning_episodes = 499
    sb = SubGoalDiscovery(env=env, name="Sub-goal Discovery stage 1", discovery_on_episodes=[learning_episodes])
    cozy = CozyRL(environment=env, agents=[
        sb,
        # SubGoalDiscovery(env=env, name="Sub-goal discovery on [300]", discovery_on_episodes=[300]),
        # SubGoalDiscovery(env=env, name="Sub-goal discovery on [300, 600]", discovery_on_episodes=[300, 600]),
        # StandardHAM(env, name="Standard"),
        # Q_Agent(env=TwoRooms(), name="Q-learning", eps=0.1),
        # Q_Agent(env=TwoRooms(), name="Q-3", eps=0.7),
        # StandardHAM(env=env, name="stHAM"),

    ],
                  supplementary=[
                      # RewardChartDrawer(smooth_step=50)
                  ],
                  number_of_episodes=learning_episodes+1)

    cozy.run()
    sb.discovery_on_episodes = []
    sb.name = "Sub-goal Discovery stage 2"
    cozy = CozyRL(environment=env, agents=[
        sb,
        # SubGoalDiscovery(env=env, name="Sub-goal discovery on [300]", discovery_on_episodes=[300]),
        # SubGoalDiscovery(env=env, name="Sub-goal discovery on [300, 600]", discovery_on_episodes=[300, 600]),
        # StandardHAM(env, name="Standard"),
        # SubGoalDiscovery(env=env, name="Standard", discovery_on_episodes=[]),
        Q_Agent(env=TwoRooms(), name="Q-3", eps=0.1),
        # StandardHAM(env=env, name="stHAM"),

    ],
                  supplementary=[RewardChartDrawer(smooth_step=10)],
                  number_of_episodes=2000)

    cozy.run()


if __name__ == '__main__':
    main()
