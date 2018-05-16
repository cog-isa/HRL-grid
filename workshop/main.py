from HAM.HAM_core import Stop, Start, Action, Choice
from environments.arm_env.arm_env import ArmEnvToggleTopOnly
from workshop.check_graphs import generate_good_graph_ids


def main():
    env = ArmEnvToggleTopOnly(size_x=5, size_y=5, cubes_cnt=4, episode_max_length=600, finish_reward=100, action_minus_reward=-0.001, tower_target_size=4)
    vertexes = sorted([
        Stop(),
        Start(),

        Action(env.ACTIONS.LEFT),
        Action(env.ACTIONS.RIGHT),
        Action(env.ACTIONS.UP),
        Action(env.ACTIONS.DOWN),
        # Action(env.ACTIONS.TOGGLE),
        Choice(),
        # Action(env.ACTIONS.LEFT),
        # Action(env.ACTIONS.RIGHT),
        # Action(env.ACTIONS.UP),
        # Action(env.ACTIONS.DOWN),
        # Action(env.ACTIONS.TOGGLE),
        # Choice(),
    ])
    with open("good_graph_id.txt", "w") as f:
        for id in generate_good_graph_ids(env=env, vertexes=vertexes):
            f.write(str(id) + " ")

    with open("good_graph_id.txt") as f:
        good_ids = list(map(int, f.read().split()))
        print(good_ids)


if __name__ == '__main__':
    main()
