from time import sleep
import pandas as pd
import numpy as np
from collections import namedtuple, defaultdict
from sklearn.preprocessing import MinMaxScaler
import sys
from gym.envs.toy_text import discrete
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from HAM.HAM_core import AbstractMachine, Start, Choice, Action, Stop, \
    MachineRelation, MachineGraph
from HAM.HAM_utils import HAMParamsCommon, PlotParams, plot_multi


class AutoMachineSimple(AbstractMachine):
    def __init__(self, env):
        start = Start()
        choice_one = Choice()
        actions = [Action(action=_) for _ in env.get_actions_as_dict().values()]
        stop = Stop()

        transitions = [MachineRelation(left=start, right=choice_one), ]
        for action in actions:
            transitions.append(MachineRelation(left=choice_one, right=action))
            transitions.append(MachineRelation(left=action, right=stop, label=0))
            transitions.append(MachineRelation(left=action, right=stop, label=1))

        super().__init__(graph=MachineGraph(transitions=transitions))


def arg_max_action(q_dict, state, action_space):
    result_action = 0
    for action_to in range(action_space):
        if q_dict[state, action_to] > q_dict[state, result_action]:
            result_action = action_to
    return result_action


class TwoRooms(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    ACTIONS = namedtuple("ACTIONS", ["LEFT", "UP", "RIGHT", "DOWN", ])(
        UP=0,
        RIGHT=1,
        DOWN=2,
        LEFT=3,
    )

    CONSTANTS = namedtuple("CONSTANTS", ["FREE_CELL", "OBSTACLE", "TARGET"])(
        FREE_CELL=0,
        OBSTACLE=1,
        TARGET=2,
    )

    MOVES = [ACTIONS.UP, ACTIONS.RIGHT, ACTIONS.DOWN, ACTIONS.LEFT]
    MOVES_X_Y = {ACTIONS.UP: (0, -1), ACTIONS.RIGHT: (1, 0), ACTIONS.DOWN: (0, 1), ACTIONS.LEFT: (-1, 0)}

    def get_actions_as_dict(self):
        return {_: getattr(self.ACTIONS, _) for _ in self.ACTIONS._fields}

    def get_current_state(self):
        return self.s

    def is_done(self):
        return self.done

    def _step(self, a):
        res = next_s, reward, done, _ = super(TwoRooms, self)._step(a)
        self.done = done
        return res

    def __init__(self):

        finish_reward = 1

        co = self.CONSTANTS

        self.code_middle = 2 ** 7

        action_mapping = {
            self.ACTIONS.RIGHT: np.array([0, 1]),
            self.ACTIONS.LEFT: np.array([0, -1]),
            self.ACTIONS.UP: np.array([-1, 0]),
            self.ACTIONS.DOWN: np.array([1, 0]),
        }

        self.mark = []

        # building 2-rooms maze
        self._maze = np.full(shape=(12, 16), fill_value=co.FREE_CELL).astype(np.int32)
        # feel boundaries of room with obstacles
        self._maze[0, :] = self._maze[:, 0] = co.OBSTACLE
        self._maze[self._maze.shape[0] - 1, :] = co.OBSTACLE
        self._maze[:, self._maze.shape[1] - 1] = co.OBSTACLE

        # separate rooms
        self._maze[:, self._maze.shape[1] // 2] = co.OBSTACLE

        # clear obstacles for door
        self._maze[self._maze.shape[0] // 2, self._maze.shape[1] // 2] = 0
        self._maze[self._maze.shape[0] // 2 - 1, self._maze.shape[1] // 2] = 0

        # placing target at the lower right corner of the right hand room
        self._maze[self._maze.shape[0] - 2, self._maze.shape[1] - 2] = co.TARGET

        prob = {}

        def append_transitions_from_cell(a_x, a_y, p):
            state = self.encode(a_x, a_y)
            p[state] = {a: [] for a in range(len(self.ACTIONS))}
            for a in self.ACTIONS:
                for a2 in self.ACTIONS:
                    dx, dy = action_mapping[a2]
                    a_n_x, a_n_y = a_x + dx, a_y + dy
                    if self._maze[a_n_x][a_n_y] == co.OBSTACLE:
                        new_state = state
                    else:
                        new_state = self.encode(a_n_x, a_n_y)
                    done = self._maze[a_n_x, a_n_y] == co.TARGET
                    reward = finish_reward if self._maze[a_n_x, a_n_y] == co.TARGET else -0.01
                    probability = 0.7 if a == a2 else 0.1
                    p[state][a].append((probability, new_state, reward, done))

        for agent_x1 in range(self._maze.shape[0]):
            for agent_y1 in range(self._maze.shape[1]):
                if self._maze[agent_x1][agent_y1] == co.OBSTACLE:
                    continue
                append_transitions_from_cell(agent_x1, agent_y1, prob)

        isd = []
        for x in range(1, self._maze.shape[0] - 1):
            for y in range(1, self._maze.shape[1] // 2):
                isd.append(self.encode(x, y))
        isd = np.array(isd)
        super(TwoRooms, self).__init__(self.encode(self._maze.shape[0] - 1, self._maze.shape[1] - 1), len(self.ACTIONS),
                                       prob, isd)

    def _reset(self):
        self.done = False
        self.s = np.random.choice(self.isd, size=1)[0]
        return self.s

    def encode(self, x, y):
        # checking constraints for x,y coordinates
        assert 0 <= x < self._maze.shape[0] and 0 <= y < self._maze.shape[1]
        # checking constraints for shape[1]
        assert self._maze.shape[1] < self.code_middle

        return x * self.code_middle + y

    def decode(self, state):
        return state // self.code_middle, state % self.code_middle

    def _render(self, mode='human', close=False, mark=None):
        if close:
            return

        outfile = sys.stdout

        maze_size_x = len(self._maze)
        maze_size_y = len(self._maze[0])
        output = "\n"
        for i in range(maze_size_x):
            for j in range(maze_size_y):
                output += " "
                if self.encode(i, j) in self.mark:
                    output += self.mark[self.encode(i, j)]
                elif self.s == self.encode(i, j):
                    output += "x"
                else:
                    if self._maze[i][j] == 0:
                        output += "."
                    if self._maze[i][j] == 1:
                        output += "H"
                    if self._maze[i][j] == 2:
                        output += "F"
                    if self._maze[i][j] == 3:
                        output += "F"
                output += " "
            output += '\n'
        outfile.write(output)


def q_learning(env, num_episodes, eps=0.1, alpha=0.1, gamma=0.9):
    to_plot = []

    q_table = defaultdict(lambda: 0)
    bns_count = defaultdict(lambda: 0)
    V = defaultdict(lambda: None)

    for _ in tqdm(range(num_episodes)):
        ep_reward = 0
        eps *= 0.9
        s = env.reset()

        bn_added = {}
        while True:
            if np.random.rand(1) < eps:
                action = np.random.choice(env.action_space.n, size=1)[0]
            else:
                action = arg_max_action(q_dict=q_table, state=s, action_space=env.action_space.n)

            next_s, reward, done, _ = env.step(action)
            a = arg_max_action(q_dict=q_table, state=s, action_space=env.action_space.n)
            # noinspection PyTypeChecker
            V[s] = (*env.decode(s), q_table[s, a])
            # making +1 to bn_counts once for each episode
            if not bn_added.get(s, False):
                bns_count[s] += 1
                bn_added[s] = True
            q_table[s, action] = (1 - alpha) * q_table[s, action] + alpha * (reward + gamma * q_table[next_s, a])

            ep_reward += reward
            if done:
                break

            s = next_s
        to_plot.append(ep_reward)
    sleep(0.1)

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
        X[["V"]] *= 0.5
        # df[["x", "y"]] = df[["x", "y"]].apply(np.float)
        df["x"] = df["x"].astype(np.float)
        df["y"] = df["y"].astype(np.float)

        sc.fit(np.vstack((df[["x"]], df[["y"]])))

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

    def runner(hams, num_episodes, env):
        for i_episode in range(1, num_episodes + 1):
            env.reset()
            while not env.is_done():
                for ham in hams:
                    if env.s in ham.states_in_my_cluster:
                        while not env.is_done() and env.s not in ham.bns:
                            ham.machine.run(params)
                        while not env.is_done() and env.s in ham.states_in_my_cluster:
                            ham.machine.run(params)

            if i_episode % 10 == 0:
                print("\r{ham} episode {i_episode}/{num_episodes}.".format(**locals()), end="")
                sys.stdout.flush()

    class BnsMachine:
        def __init__(self, params, cluster_index, list_of_bns, states_in_my_cluster):
            self.machine = AutoMachineSimple(env)
            self.cluster_index = cluster_index
            self.bns = set(list_of_bns)
            self.states_in_my_cluster = states_in_my_cluster
            self.params = params

    params = HAMParamsCommon(env)
    hams = [BnsMachine(params=params, cluster_index=_, list_of_bns=map_cluster_to_sorted_bns[_][:BNS_FOR_CLUSTER],
                       states_in_my_cluster=set(map_cluster_to_sorted_bns[_])) for _ in
            map_cluster_to_sorted_bns]

    runner(hams=hams,
           num_episodes=500,
           env=env,
           )
    to_plot = list()
    to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="HAM_with_pull_up"))
    plot_multi(to_plot)
    # print(params.logs["ep_rewards"])
    return to_plot, q_table


def main():
    q_s, q_t = q_learning(TwoRooms(), 500)


if __name__ == '__main__':
    main()
