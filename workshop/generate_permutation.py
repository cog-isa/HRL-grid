from itertools import combinations

from HAM.HAM_core import Start, Stop, Action, Call, Choice
from environments.arm_env.arm_env import ArmEnvToggleTopOnly


def generate_list_of_vertexes(vertex_types, vertex_of_each_type_max_count, max_vertex_count):
    pass


def main():
    env = ArmEnvToggleTopOnly(size_x=5, size_y=5, cubes_cnt=4, episode_max_length=600, finish_reward=100, action_minus_reward=-0.001, tower_target_size=4)
    vertex_types = sorted([
        Start(),
        Stop(),
        Action(env.ACTIONS.LEFT),
        Action(env.ACTIONS.LEFT),
        Call(None),
        Choice(),
        Action(env.ACTIONS.RIGHT),
        Action(env.ACTIONS.TOGGLE),
        Action(env.ACTIONS.TOGGLE),
    ])
    generate_list_of_vertexes(vertex_types=vertex_types, vertex_of_each_type_max_count=3, max_vertex_count=7)
    if __name__ == '__main__':
        main()
