from HAM_new.HAM_core import AutoBasicMachine, RootMachine, LoopInvokerMachine, AbstractMachine, Start, Choice, Action, Stop, MachineRelation, Call, \
    MachineGraph
from HAM_new.HAM_experiments.HAM_utils import HAMParamsCommon, maze_world_input_01, plot_multi, ham_runner, PlotParams
from environments.arm_env.arm_env import ArmEnv

to_plot = []
env = ArmEnv(episode_max_length=300,
             size_x=5,
             size_y=3,
             cubes_cnt=4,
             action_minus_reward=-1,
             finish_reward=100,
             tower_target_size=4)
num_episodes = 300

params = HAMParamsCommon(env)
ham_runner(ham=AutoBasicMachine(env), num_episodes=num_episodes, env=env, params=params)
to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="HAM_basic"))

# --------------------------------------------------------

pull_up_start = Start()
pull_up_on = Action(action=env.get_actions_as_dict()["ON"])
pull_up_down_01 = Action(action=env.get_actions_as_dict()["DOWN"])
pull_up_down_02 = Action(action=env.get_actions_as_dict()["DOWN"])
pull_up_down_03 = Action(action=env.get_actions_as_dict()["DOWN"])
pull_up_down_04 = Action(action=env.get_actions_as_dict()["DOWN"])
pull_up_up_01 = Action(action=env.get_actions_as_dict()["UP"])
pull_up_up_02 = Action(action=env.get_actions_as_dict()["UP"])
pull_up_up_03 = Action(action=env.get_actions_as_dict()["UP"])
pull_up_up_04 = Action(action=env.get_actions_as_dict()["UP"])
pull_up_stop = Stop()

pull_up_transitions = (
    MachineRelation(left=pull_up_start, right=pull_up_on),

    MachineRelation(left=pull_up_on, right=pull_up_down_01, label=0),
    MachineRelation(left=pull_up_down_01, right=pull_up_down_02, label=0),
    MachineRelation(left=pull_up_down_02, right=pull_up_down_03, label=0),
    MachineRelation(left=pull_up_down_03, right=pull_up_down_04, label=0),
    MachineRelation(left=pull_up_down_04, right=pull_up_up_01, label=0),
    MachineRelation(left=pull_up_up_01, right=pull_up_up_02, label=0),
    MachineRelation(left=pull_up_up_02, right=pull_up_up_03, label=0),
    MachineRelation(left=pull_up_up_03, right=pull_up_up_04, label=0),
    MachineRelation(left=pull_up_up_04, right=pull_up_stop, label=0),

    MachineRelation(left=pull_up_on, right=pull_up_stop, label=1),
    MachineRelation(left=pull_up_down_01, right=pull_up_stop, label=1),
    MachineRelation(left=pull_up_down_02, right=pull_up_stop, label=1),
    MachineRelation(left=pull_up_down_03, right=pull_up_stop, label=1),
    MachineRelation(left=pull_up_down_04, right=pull_up_stop, label=1),
    MachineRelation(left=pull_up_up_01, right=pull_up_stop, label=1),
    MachineRelation(left=pull_up_up_02, right=pull_up_stop, label=1),
    MachineRelation(left=pull_up_up_03, right=pull_up_stop, label=1),
    MachineRelation(left=pull_up_up_04, right=pull_up_stop, label=1),
)
pull_up = AbstractMachine(MachineGraph(transitions=pull_up_transitions))

start = Start()
choice_one = Choice()
left = Action(action=env.get_actions_as_dict()["LEFT"])
right = Action(action=env.get_actions_as_dict()["RIGHT"])
off = Action(action=env.get_actions_as_dict()["OFF"])

call = Call(machine_to_call=pull_up)

stop = Stop()

transitions = (
    MachineRelation(left=start, right=choice_one),
    MachineRelation(left=choice_one, right=left),
    MachineRelation(left=choice_one, right=right),
    MachineRelation(left=choice_one, right=off),
    MachineRelation(left=choice_one, right=call),

    MachineRelation(left=call, right=stop),

    MachineRelation(left=left, right=stop, label=0),
    MachineRelation(left=right, right=stop, label=0),
    MachineRelation(left=off, right=stop, label=0),

    MachineRelation(left=left, right=stop, label=1),
    MachineRelation(left=right, right=stop, label=1),
    MachineRelation(left=off, right=stop, label=1),
)

pull_up_machine = RootMachine(machine_to_invoke=LoopInvokerMachine(AbstractMachine(MachineGraph(transitions=transitions))))

params = HAMParamsCommon(env)
ham_runner(ham=pull_up_machine, num_episodes=num_episodes, env=env, params=params)
to_plot.append(PlotParams(curve_to_draw=params.logs["ep_rewards"], label="HAM_with_pull_up"))

plot_multi(to_plot)
