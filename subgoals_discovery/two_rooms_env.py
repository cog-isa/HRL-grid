import numpy as np
from collections import namedtuple
import sys
from gym.envs.toy_text import discrete

from HAM.HAM_core import AbstractMachine, Start, Stop, \
    MachineRelation, MachineGraph, HAMParams, ActionSimple, ChoiceSimple


class AutoMachineSimple(AbstractMachine):
    def __init__(self, env):
        start = Start()
        choice_one = ChoiceSimple()
        actions = [ActionSimple(action=_) for _ in env.get_actions_as_dict().values()]
        stop = Stop()

        transitions = [MachineRelation(left=start, right=choice_one), ]
        for action in actions:
            transitions.append(MachineRelation(left=choice_one, right=action))
            transitions.append(MachineRelation(left=action, right=stop, label=0))
            transitions.append(MachineRelation(left=action, right=stop, label=1))

        self.last_action_vertex = None
        super().__init__(graph=MachineGraph(transitions=transitions))

    def run(self, params: HAMParams):
        t = filter(lambda x: isinstance(x.left, Start), self.graph.transitions)
        try:
            current_vertex = t.__next__().left
        except StopIteration:
            raise Exception("No start vertex in graph")
        try:
            t.__next__()
            raise Exception("More than one start vertex in graph")
        except StopIteration:
            pass

        self.params = params
        # shortcut lambda for on_model function
        self.get_on_model_transition_id = lambda: self.params.on_model_transition_id_function(self.params.env)
        while not isinstance(current_vertex, Stop):
            if isinstance(current_vertex, ActionSimple) and current_vertex.action is not None:
                self.last_action_vertex = current_vertex
                # current_vertex = current_vertex.run(self)

                return current_vertex.action
            current_vertex = current_vertex.run(self)

    def update_after_action(self, reward):
        if self.last_action_vertex is None:
            raise KeyError
        current_vertex = self.last_action_vertex.run(self, reward)
        while not isinstance(current_vertex, Stop):
            current_vertex = current_vertex.run()


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

        finish_reward = 100
        dangerous_state_reward = -20
        minus_reward = -0.1
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
        self._maze = np.full(shape=(30, 30), fill_value=co.FREE_CELL).astype(np.int32)
        # feel boundaries of room with obstacles

        self._maze[0, :] = self._maze[:, 0] = co.OBSTACLE
        self._maze[self._maze.shape[0] - 1, :] = co.OBSTACLE
        self._maze[:, self._maze.shape[1] - 1] = co.OBSTACLE

        # separate rooms
        self._maze[:, self._maze.shape[1] // 2] = co.OBSTACLE
        self._maze[self._maze.shape[0] // 2, :] = co.OBSTACLE

        self._maze[self._maze.shape[0] // 2, self._maze.shape[1] // 4] = co.FREE_CELL
        self._maze[self._maze.shape[0] // 2, self._maze.shape[1] // 4 * 3] = co.FREE_CELL
        self._maze[self._maze.shape[0] // 4, self._maze.shape[1] // 2] = co.FREE_CELL
        # self._maze[self._maze.shape[0] // 4 * 3, self._maze.shape[1] // 2] = co.FREE_CELL
        # clear obstacles for door
        # self._maze[self._maze.shape[0] // 2, self._maze.shape[1] // 2] = 0
        # self._maze[self._maze.shape[0] // 2 - 1, self._maze.shape[1] // 2] = 0

        # placing target at the lower right corner of the right hand room
        self._maze[self._maze.shape[0] - 2, self._maze.shape[1] - 2] = co.TARGET

        prob = {}

        self.dangerous_state = {
            self.encode(self._maze.shape[0] // 4, self._maze.shape[1] // 4),
            self.encode(self._maze.shape[0] // 4 * 3+1, self._maze.shape[1] // 4),
            self.encode(self._maze.shape[0] // 4, (self._maze.shape[1] + 1) // 4 * 3),
            self.encode(self._maze.shape[0] // 4 * 3+1, self._maze.shape[1] // 4 * 3),

            # self.encode(8, self._maze.shape[1] // 4 * 3),
            # self.encode(4, self._maze.shape[1] // 4),
            # self.encode(8, self._maze.shape[1] // 4 * 3),
            # self.encode(10, 12),
        }

        def append_transitions_from_cell(a_x, a_y, p):
            state = self.encode(a_x, a_y)
            p[state] = {a: [] for a in range(len(self.ACTIONS))}
            # print(a_x, a_y)
            for a in self.ACTIONS:
                for a2 in self.ACTIONS:
                    dx, dy = action_mapping[a2]
                    a_n_x, a_n_y = a_x + dx, a_y + dy
                    if self._maze[a_n_x][a_n_y] == co.OBSTACLE:
                        new_state = state
                    else:
                        new_state = self.encode(a_n_x, a_n_y)
                    done = self._maze[a_n_x, a_n_y] == co.TARGET
                    reward = finish_reward if self._maze[a_n_x, a_n_y] == co.TARGET else minus_reward
                    probability = 0.91 if a == a2 else 0.03
                    # probability = 1.0 if a == a2 else 0.0
                    if new_state in self.dangerous_state:
                        reward = dangerous_state_reward
                        done = True

                    p[state][a].append((probability, new_state, reward, done))

        for agent_x1 in range(self._maze.shape[0]):
            for agent_y1 in range(self._maze.shape[1]):
                if self._maze[agent_x1][agent_y1] == co.OBSTACLE:
                    continue
                append_transitions_from_cell(agent_x1, agent_y1, prob)

        isd = []
        isd.append(self.encode(self._maze.shape[0] // 2 + 1, 2))
        for x in range(self._maze.shape[0] // 2 + 1, self._maze.shape[0] - 1):
            for y in range(1, self._maze.shape[1] // 2):
                if self.encode(x, y) not in self.dangerous_state:
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
                if self.encode(i, j) in self.dangerous_state:
                    output += "D"
                elif self._maze[i][j] == 2:
                    output += "F"
                elif self.encode(i, j) in self.mark:
                    output += self.mark[self.encode(i, j)]
                elif self.s == self.encode(i, j):
                    output += "x"
                else:
                    if self._maze[i][j] == 0:
                        output += "."
                    if self._maze[i][j] == 1:
                        output += "H"

                    if self._maze[i][j] == 3:
                        output += "F"
                output += " "
            output += '\n'
        outfile.write(output)
