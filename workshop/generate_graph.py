import re

from HAM.HAM_core import Action, MachineRelation, Stop, Start, AbstractMachine, MachineGraph, Call, Choice
from environments.arm_env.arm_env import ArmEnvToggleTopOnly
from utils.graph_drawer import draw_graph


class MachineStored:
    def __init__(self, vertex_types, binary_matrix_representation, env):
        # self.vertex_types = sorted(vertex_types)
        self.vertex_types = vertex_types
        self.binary_matrix_representation = binary_matrix_representation
        self.env = env

        for i in range(len(self.vertex_types) - 1):
            assert not self.vertex_types[i + 1] < self.vertex_types[i], "should be sorted"

    def __str__(self):
        return ""

    def get_machine(self):
        transitions = []
        for left_ind in range(len(self.vertex_types)):
            for right_ind in range(len(self.vertex_types)):
                left = self.vertex_types[left_ind]
                right = self.vertex_types[right_ind]
                if (2 ** (left_ind * len(self.vertex_types) + right_ind)) & self.binary_matrix_representation:
                    if isinstance(left, Action):
                        transitions.append(MachineRelation(left=left, right=right, label=0))
                    else:
                        transitions.append(MachineRelation(left=left, right=right))

        start, stop = None, None
        for vertex in self.vertex_types:
            if isinstance(vertex, Start):
                start = vertex
            elif isinstance(vertex, Stop):
                stop = vertex

        assert start is not None
        assert stop is not None

        for vertex in [_ for _ in self.vertex_types if isinstance(_, Action)]:
            transitions.append(MachineRelation(left=vertex, right=stop, label=1))

        return AbstractMachine(MachineGraph(transitions=transitions))

    def get_max_index(self):
        return 2 ** (len(self.vertex_types) ** 2)

    def draw(self, filename):
        draw_graph(filename, self.get_machine().get_graph_to_draw(action_to_name_mapping=self.env.get_actions_as_dict(), no_edges_with_exit_f=True))
        s = None
        with open("{filename}.svg".format(**locals()), "r") as f:
            s = f.readlines()
            s = [re.sub(r"Action\d+", r"Action", _) for _ in s]
            s = [re.sub(r"Choice\d+", r"Choice", _) for _ in s]
            s = [re.sub(r"Call\d+", r"Call", _) for _ in s]
            s = [re.sub(r"Stop\d+", r"Stop", _) for _ in s]
            s = [re.sub(r"Start\d+", r"Start", _) for _ in s]
        with open("{filename}.svg".format(**locals()), "w") as f:
            f.writelines(s)

    def draw_ru(self, filename):
        action_to_name_mapping = self.env.get_actions_as_dict()
        ru_mapping = {"LEFT": "ВЛЕВО",
                      "RIGHT": "ВПРАВО",
                      "DOWN": "ВНИЗ",
                      "TOGGLE": "ПЕРЕКЛ.",
                      "UP": "ВВЕРХ"
                      }

        action_to_name_mapping_ru = {

        }
        for key in action_to_name_mapping.keys():
            assert key in ru_mapping, "don't worry you just should add translation of key <<{key}>> to ru_mapping dict placed above".format(**locals())
            action_to_name_mapping_ru[ru_mapping[key]] = action_to_name_mapping[key]

        draw_graph(filename, self.get_machine().get_graph_to_draw(action_to_name_mapping=action_to_name_mapping_ru, no_edges_with_exit_f=True))
        s = None

        with open("{filename}.svg".format(**locals()), "r") as f:
            s = f.readlines()
            s = [re.sub(r"Action\d+", r"Действие", _) for _ in s]
            s = [re.sub(r"Choice\d+", r"Выбор", _) for _ in s]
            s = [re.sub(r"Call\d+", r"Вызов", _) for _ in s]
            s = [re.sub(r"Stop\d+", r"Стоп", _) for _ in s]
            s = [re.sub(r"Start\d+", r"Старт", _) for _ in s]
        with open("{filename}.svg".format(**locals()), "w") as f:
            f.writelines(s)


def main():
    pass
    env = ArmEnvToggleTopOnly(size_x=5, size_y=5, cubes_cnt=4, episode_max_length=600, finish_reward=100, action_minus_reward=-0.001, tower_target_size=4)

    ms = MachineStored(vertex_types=sorted([
        Start(),
        Stop(),
        Action(env.ACTIONS.LEFT),
        Action(env.ACTIONS.LEFT),
        Call(None),
        Choice(),
        Action(env.ACTIONS.RIGHT),
        Action(env.ACTIONS.TOGGLE),
        Action(env.ACTIONS.TOGGLE),
    ]), binary_matrix_representation=543215435343, env=env)
    for i in range(100):
        ms.binary_matrix_representation = i
        ms.draw_ru("ololo{i}".format(**locals()))


if __name__ == '__main__':
    main()
