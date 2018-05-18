import json

from HAM.HAM_core import Stop, Start, Action, Choice, AbstractMachine, MachineRelation, MachineGraph
from HAM.HAM_experiments.HAM_utils import HAMParamsCommon, plot_multi, PlotParams
from environments.arm_env.arm_env import ArmEnvToggleTopOnly
from workshop.check_graphs import generate_good_graph_ids, generate_machines_by_ids, generate_good_graphs
from workshop.generate_graph import MachineStored
import sys


def compress_graphs_dicts(g_list):
    ss = set()
    res = []
    for item in g_list:
        item_json = json.dumps(obj=item)
        if item_json in ss:
            continue
        else:
            res.append(item)
            ss.add(item_json)

    return res


def part_one(env, vertexes):
    t = compress_graphs_dicts([_.to_dict() for _ in generate_good_graphs(env=env, vertexes=vertexes, vertex_count=3)])
    with open("machines_part_one.json", "w") as out_f:
        json.dump(t, fp=out_f, sort_keys=True, indent=4)


def part_two(env):
    with open("machines_part_one.json") as json_file:
        machines = [MachineStored.ms_from_machine(AutoMachineSimple(env), env)]
        machines_to_save = []
        for ms_dict in json.load(json_file):
            machines.append(MachineStored.from_dict(graph_dict=ms_dict, env=env))

        m_id = 0

        params = HAMParamsCommon(env)
        am = AutoMachineSimple(env)

        runner(ham=am,
               num_episodes=2000,
               env=env,
               params=params,
               on_model_mapping={},
               )
        qv = params.q_value

        for on_model_part in list(reversed(env.get_all_on_model())):
            for ms in machines:
                machine = ms.get_machine()

                params = HAMParamsCommon(env)
                params.q_value = qv

                runner(ham=am,
                       num_episodes=1,
                       env=env,
                       params=params,
                       on_model_mapping={on_model_part: machine},
                       )
                to_plot = list()
                to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="HAM_with_pull_up"))
                total_reward = sum(params.logs["ep_rewards"])
                print("rewards sum:", total_reward)
                # plot_multi(to_plot, filename="pics/" + str(m_id) + ":::" + str(on_model_part) + ":::" + str(ms.binary_matrix_representation) + ":::" + str(sum(params.logs["ep_rewards"])))
                # ms.draw("pics/" + str(m_id) + ":" + str(ms.binary_matrix_representation) + ":" + str(total_reward))
                m_id += 1

                if total_reward > 10:
                    machines_to_save.append(ms)
        with open("machines_part_two.json", "w") as out_f:
            t = compress_graphs_dicts([_.to_dict() for _ in machines_to_save])
            json.dump(obj=t, fp=out_f, sort_keys=True, indent=4)


def part_three(env):
    with open("machines_part_two.json") as json_file:
        machines = []
        for ms_dict in json.load(json_file):
            machines.append(MachineStored.from_dict(graph_dict=ms_dict, env=env))

        cluster_best_result_mapper = {}
        cluster_best_machine_mapper = {}

        for on_model_part in list(reversed(env.get_all_on_model())):
            for ms in machines:
                machine = ms.get_machine()

                params = HAMParamsCommon(env)

                runner(ham=AutoMachineSimple(env),
                       num_episodes=300,
                       env=env,
                       params=params,
                       on_model_mapping={on_model_part: machine},
                       no_output=True,
                       )
                to_plot = list()
                to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="HAM_with_pull_up"))
                total_reward = sum(params.logs["ep_rewards"])
                print(total_reward)
                on_model_part_str = str(on_model_part)
                if on_model_part_str in cluster_best_result_mapper:
                    if cluster_best_result_mapper[on_model_part_str] < total_reward:
                        cluster_best_result_mapper[on_model_part_str], cluster_best_machine_mapper[on_model_part_str] = total_reward, ms.to_dict()
                else:
                    cluster_best_result_mapper[on_model_part_str], cluster_best_machine_mapper[on_model_part_str] = total_reward, ms.to_dict()

        with open("machines_part_three.json", "w") as out_f:
            json.dump(obj=cluster_best_machine_mapper, fp=out_f, sort_keys=True, indent=4)


def part_four(env):
    with open("machines_part_three.json") as json_file:
        cluster_best_machine_mapper_str_key = json.load(json_file)
        cluster_best_machine_mapper = {}
        for key in cluster_best_machine_mapper_str_key:
            tuple_key = key
            tuple_key = tuple_key.replace("(", "")
            tuple_key = tuple_key.replace(")", "")
            tuple_key = tuple(map(eval, tuple_key.split(",")))
            cluster_best_machine_mapper[tuple_key] = MachineStored.from_dict(cluster_best_machine_mapper_str_key[key], env=env).get_machine()

        params = HAMParamsCommon(env)

        runner(ham=AutoMachineSimple(env),
               num_episodes=300,
               env=env,
               params=params,
               on_model_mapping=cluster_best_machine_mapper,
               no_output=True,
               )
        to_plot = list()
        to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="HAM_with_pull_up"))
        total_reward = sum(params.logs["ep_rewards"])
        print("united:", total_reward)
        plot_multi(to_plot, filename="diagrampics")


def part_five(env):
    with open("machines_part_three.json") as json_file:
        cluster_best_machine_mapper_str_key = json.load(json_file)
        cluster_best_machine_mapper = {}
        for key in cluster_best_machine_mapper_str_key:
            tuple_key = key
            tuple_key = tuple_key.replace("(", "")
            tuple_key = tuple_key.replace(")", "")
            tuple_key = tuple(map(eval, tuple_key.split(",")))
            cluster_best_machine_mapper[tuple_key] = MachineStored.from_dict(cluster_best_machine_mapper_str_key[key], env=env).get_machine()

        params = HAMParamsCommon(env)

        runner(ham=AutoMachineSimple(env),
               num_episodes=5000,
               env=env,
               params=params,
               on_model_mapping=cluster_best_machine_mapper,
               no_output=True,
               )
        to_plot = list()
        to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="clustering"))


        params = HAMParamsCommon(env)

        runner(ham=AutoMachineSimple(env),
               num_episodes=5000,
               env=env,
               params=params,
               on_model_mapping={},
               no_output=True,
               )
        to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="q-learning"))

        plot_multi(to_plot, filename="b")


def runner(ham, num_episodes, env, params, on_model_mapping, no_output=None, ):
    for i_episode in range(1, num_episodes + 1):
        env.reset()

        while not env.is_done():
            # print(env.get_on_model())
            if env.get_on_model() in on_model_mapping:
                on_model_mapping[env.get_on_model()].run(params)
            else:
                ham.run(params)
        # env.render()
        if i_episode % 10 == 0:
            if no_output is None:
                print("\r{ham} episode {i_episode}/{num_episodes}.".format(**locals()), end="")
                sys.stdout.flush()


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


def main():
    def get_on_model(self):
        return self.get_arm_x(), self.is_cube_graped()

    def get_all_on_model(self):
        res = []
        for height in range(1, self._size_x + 1):
            for graped in [True, False]:
                res.append((height, graped))
        return res

    def get_arm_x(self):
        return self._size_x - self._arm_x

    def is_cube_graped(self):
        cube_dx, cube_dy = self.MOVE_ACTIONS[self.ACTIONS.DOWN]
        cube_x, cube_y = self._arm_x + cube_dx, self._arm_y + cube_dy
        return self._magnet_toggle and self.ok(cube_x, cube_y) and self._grid[cube_x][cube_y] == 1

    ArmEnvToggleTopOnly.get_arm_x = get_arm_x
    ArmEnvToggleTopOnly.get_all_on_model = get_all_on_model
    ArmEnvToggleTopOnly.is_cube_graped = is_cube_graped
    ArmEnvToggleTopOnly.get_on_model = get_on_model

    # env = ArmEnvToggleTopOnly(size_x=5, size_y=4, cubes_cnt=4, episode_max_length=500, finish_reward=100, action_minus_reward=-0.001, tower_target_size=4)
    env = ArmEnvToggleTopOnly(size_x=4, size_y=3, cubes_cnt=3, episode_max_length=300, finish_reward=100, action_minus_reward=-0.00001, tower_target_size=3)

    vertexes = sorted([
        Stop(),
        Start(),

        Action(env.ACTIONS.LEFT),
        Action(env.ACTIONS.RIGHT),
        Action(env.ACTIONS.UP),
        Action(env.ACTIONS.DOWN),
        Action(env.ACTIONS.TOGGLE),
        Choice(),
        Action(env.ACTIONS.LEFT),
        Action(env.ACTIONS.RIGHT),
        Action(env.ACTIONS.UP),
        Action(env.ACTIONS.DOWN),
        Action(env.ACTIONS.TOGGLE),
        # Choice(),
    ])

    # part_one(env, vertexes)
    # part_two(env)
    # part_three(env)
    # part_four(env)


    env = ArmEnvToggleTopOnly(size_x=5, size_y=4, cubes_cnt=4, episode_max_length=300, finish_reward=100, action_minus_reward=-0.00001, tower_target_size=4)
    part_five(env)


if __name__ == '__main__':
    main()
