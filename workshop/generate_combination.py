from copy import deepcopy

from HAM.HAM_core import Start, Stop, Action, Call, Choice
from environments.arm_env.arm_env import ArmEnvToggleTopOnly


def vertex_list_to_str(vertex_list):
    ans = ""
    for i in vertex_list:
        ans = ans + " " + str(i)
    return ans


def generate_list_of_vertexes(vertex_types, vertex_of_each_type_max_count, max_vertex_count, current_list=None, deep=0, ans=None):
    if current_list is None:
        current_list = []
    if ans is None:
        ans = []
    if max_vertex_count == len(current_list):
        ans.append(deepcopy(current_list))
        return ans

    if deep < len(vertex_types):
        for add_times in range(0, vertex_of_each_type_max_count + 1):
            for _ in range(add_times):
                current_list.append(vertex_types[deep])
            generate_list_of_vertexes(vertex_types, vertex_of_each_type_max_count, max_vertex_count, current_list, deep=deep + 1, ans=ans)
            for _ in range(add_times):
                current_list.pop()
    return ans


def vertex_combination(vertex_types, max_vertex_count):
    start, stop = None, None
    for vertex in vertex_types:
        if isinstance(vertex, Start):
            start = vertex
        if isinstance(vertex, Stop):
            stop = vertex

    assert start is not None and stop is not None, "Start and Stop vertex should be presented"
    assert isinstance(vertex_types[0], Start), "Start vertex should be sorted as first"
    assert isinstance(vertex_types[1], Stop), "Stop vertex should be sorted as second"

    return generate_list_of_vertexes(vertex_types=vertex_types, vertex_of_each_type_max_count=1, max_vertex_count=max_vertex_count, current_list=[start, stop],
                                     deep=2)


def main():
    env = ArmEnvToggleTopOnly(size_x=5, size_y=5, cubes_cnt=4, episode_max_length=600, finish_reward=100, action_minus_reward=-0.001, tower_target_size=4)
    vertex_types = sorted([
        Stop(),
        Start(),
        Action(env.ACTIONS.LEFT),
        Action(env.ACTIONS.RIGHT),
        Action(env.ACTIONS.UP),
        Action(env.ACTIONS.DOWN),

        Call(None),
        Choice(),
    ])
    res = generate_list_of_vertexes(vertex_types=vertex_types, vertex_of_each_type_max_count=3, max_vertex_count=5)
    for i in res:
        print(vertex_list_to_str(i))


if __name__ == '__main__':
    main()
