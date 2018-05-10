import sys
from random import randrange

from HAM.HAM_core import RootMachine, Start, Choice, Action, Stop, MachineRelation, LoopInvokerMachine, AbstractMachine, MachineGraph
from HAM.HAM_experiments.HAM_utils import HAMParamsCommon, PlotParams, plot_multi
from environments.arm_env.arm_env import ArmEnvToggleTopOnly


def main():
    def get_on_model(self):
        return self.get_arm_x(), self.is_cube_graped()

    def get_arm_x(self):
        return self._size_x - self._arm_x

    def is_cube_graped(self):
        cube_dx, cube_dy = self.MOVE_ACTIONS[self.ACTIONS.DOWN]
        cube_x, cube_y = self._arm_x + cube_dx, self._arm_y + cube_dy
        return self._magnet_toggle and self.ok(cube_x, cube_y) and self._grid[cube_x][cube_y] == 1

    ArmEnvToggleTopOnly.get_arm_x = get_arm_x
    ArmEnvToggleTopOnly.is_cube_graped = is_cube_graped
    ArmEnvToggleTopOnly.get_on_model = get_on_model

    env = ArmEnvToggleTopOnly(size_x=5, size_y=5, cubes_cnt=4, episode_max_length=600, finish_reward=100, action_minus_reward=-0.001, tower_target_size=4)

    params = HAMParamsCommon(env)
    runner(ham=AutoMachineNoLoop(env),
           num_episodes=2000,
           env=env,
           params=params,
           # no_output=True
           )
    to_plot = []
    to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="HAM_with_pull_up"))
    plot_multi(to_plot)


def runner(ham, num_episodes, env, params, no_output=None):
    ham2 = AutoMachineNoLoop(env)
    params2 = HAMParamsCommon(env)
    for i_episode in range(1, num_episodes + 1):

        env.reset()
        print("****" * 10)
        while not env.is_done():
            print(env.get_on_model())
            if i_episode % 10 >= 5:
                ham.run(params)
            else:
                pass
                ham2.run(params2)
            # print(params.previous_machine_choice_state)
        env.render()
        assert env.is_done(), "The machine is STOPPED before STOP(done) of the environment"
        if i_episode % 10 == 0:
            if no_output is None:
                print("\r{ham} episode {i_episode}/{num_episodes}.".format(**locals()), end="")
                sys.stdout.flush()


class AutoMachineNoLoop(RootMachine):
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

        super().__init__(machine_to_invoke=AbstractMachine(MachineGraph(transitions=transitions)))


if __name__ == '__main__':
    main()
