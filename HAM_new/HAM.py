import random
import operator
import sys

from environments.arm_env import ArmEnv
from lib import plotting


class HAMParams:
    def __init__(self,
                 q_value,
                 env,
                 current_state,
                 eps,
                 gamma,
                 alpha,
                 string_prefix_of_machine,
                 accumulated_discount,
                 accumulated_rewards,
                 previous_machine_choice_state,
                 env_is_done,
                 logs,
                 on_model_transition_id_function
                 ):
        self.q_value = q_value
        self.env = env
        self.current_state = current_state
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.string_prefix_of_machine = string_prefix_of_machine
        self.accumulated_discount = accumulated_discount
        self.accumulated_rewards = accumulated_rewards
        self.previous_machine_choice_state = previous_machine_choice_state
        self.env_is_done = env_is_done
        self.logs = logs
        self.on_model_transition_id_function = on_model_transition_id_function


class AbstractMachine:
    free_id = 1

    def __init__(self, transitions):
        self.transitions = transitions

        self.vertex_mapping = {_: [i for i in self.transitions if i.left == _] for _ in set([i.left for i in transitions])}

        self.action_vertex_label_mapping = {_: {__.label: __ for __ in self.vertex_mapping[_]} for _ in
                                            set([i.left for i in transitions if isinstance(i.left, Action)])}

        self.params = None
        self.get_on_model_transition_id = None

        self.previous_choice_state = None
        self.accumulated_discount = 1
        self.accumulated_rewards = 0

        # set unique id for AbstractMachine object
        self.id, AbstractMachine.free_id = AbstractMachine.free_id, AbstractMachine.free_id + 1

    def run(self, params: HAMParams):
        t = filter(lambda x: isinstance(x.left, Start), self.transitions)
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
        # TODO to think of making the code properly
        self.get_on_model_transition_id = lambda: self.params.on_model_transition_id_function(self.params.env)
        current_vertex.run(self)
        # TODO to think about to uncomment the code
        # self.params = None
        # self.get_on_model_transition_id = None


class RootMachine(AbstractMachine):
    def __init__(self, machine_to_invoke):
        start = Start()
        call = Call(machine_to_invoke)
        choice = Choice()
        stop = Stop()
        transitions = (
            MachineRelation(left=start, right=call),
            MachineRelation(left=call, right=choice),
            MachineRelation(left=choice, right=stop)
        )
        super().__init__(transitions=transitions)


class LoopInvokerMachine(AbstractMachine):
    def __init__(self, machine_to_invoke):
        start = Start()
        call = Call(machine_to_invoke)
        stop = Stop()
        empty_action = Action()
        transitions = (
            MachineRelation(left=start, right=call),
            MachineRelation(left=call, right=empty_action),
            MachineRelation(left=empty_action, right=call, label=0),
            MachineRelation(left=empty_action, right=stop, label=1),
        )
        super().__init__(transitions=transitions)


class MachineVertex:
    def __str__(self):
        return self.__class__.__name__

    def run(self, *args, **kwargs):
        raise NotImplementedError


class Start(MachineVertex):
    def run(self, own_machine: AbstractMachine):
        # calling next vertex
        return own_machine.vertex_mapping[self][0].right.run(own_machine)


class Stop(MachineVertex):
    def run(self, own_machine: AbstractMachine):
        # print(self)
        pass


class Choice(MachineVertex):
    free_id = 1

    def __init__(self):
        super(MachineVertex, self).__init__()

        # set unique id for Choice:MachineVertex object
        self.id, Choice.free_id = Choice.free_id, Choice.free_id + 1
        # print(self.__class__.__name__, self.id)

    @staticmethod
    def get_e_greedy(q_choices: dict, eps: float):
        if random.random() < eps:
            return random.choice(list(q_choices.keys()))
        else:
            return max(q_choices.items(), key=operator.itemgetter(1))[0]

    def run(self, own_machine: AbstractMachine):
        # TODO refactor foo with proper name
        foo = own_machine.id, self.id, own_machine.params.env.get_current_state()

        # TODO move the choice_relations logic to own_machine
        choice_relations = {_.id: _ for _ in own_machine.vertex_mapping[self]}

        if foo not in own_machine.params.q_value:
            own_machine.params.q_value[foo] = {_: 0 for _ in choice_relations.keys()}

        if own_machine.params.previous_machine_choice_state is not None:
            q = own_machine.params.q_value[own_machine.params.previous_machine_choice_state][own_machine.params.previous_machine_choice]
            V = own_machine.params.q_value[foo][self.get_e_greedy(own_machine.params.q_value[foo], eps=0)]
            delta = own_machine.params.alpha * (own_machine.params.accumulated_rewards + own_machine.params.accumulated_discount * V - q)
            q += delta
            own_machine.params.q_value[own_machine.params.previous_machine_choice_state][own_machine.params.previous_machine_choice] = q

        action = self.get_e_greedy(own_machine.params.q_value[foo], eps=own_machine.params.eps)
        own_machine.params.previous_machine_choice_state = foo
        own_machine.params.previous_machine_choice = action

        own_machine.params.accumulated_rewards = 0
        own_machine.params.accumulated_discount = 1

        # TODO rewrite with non-recursive variant
        return choice_relations[action].right.run(own_machine)


class Call(MachineVertex):
    def __init__(self, machine_to_call: AbstractMachine):
        super(MachineVertex, self).__init__()
        self.machine_to_call = machine_to_call

    def run(self, own_machine: AbstractMachine):
        self.machine_to_call.run(own_machine.params)

        # calling next vertex
        return own_machine.vertex_mapping[self][0].right.run(own_machine)


class Action(MachineVertex):
    def __init__(self, action=None):
        self.action = action

    def __str__(self):
        return super(Action, self).__str__() + "(" + str(self.action) + ")"

    def run(self, own_machine: AbstractMachine):
        if self.action is not None:
            state, reward, done, _ = own_machine.params.env.step(self.action)
            own_machine.params.logs["reward"] += reward
            if done:
                own_machine.params.logs["ep_rewards"].append(own_machine.params.logs["reward"])
                own_machine.params.logs["reward"] = 0

            # own_machine.params.env.render()
            own_machine.params.accumulated_rewards += reward * own_machine.params.accumulated_discount
            own_machine.params.accumulated_discount *= own_machine.params.gamma

        # calling next vertex
        return own_machine.action_vertex_label_mapping[self][own_machine.get_on_model_transition_id()].right.run(own_machine)


class MachineRelation:
    free_id = 1

    def __init__(self, left, right, label=None):
        assert not (not isinstance(left, Action) and label is not None), "Action state vertex doesn't have specified label"
        assert not (isinstance(left, Action) and label is None), "Non action state vertex has specified label"

        self.left = left
        self.right = right
        self.label = label

        # set unique id for MachineRelation object
        self.id, MachineRelation.free_id = MachineRelation.free_id, MachineRelation.free_id + 1

    def __str__(self):
        return str(self.left) + " -> " + str(self.right)


def main():
    env = ArmEnv(episode_max_length=300,
                 size_x=5,
                 size_y=4,
                 cubes_cnt=4,
                 action_minus_reward=-1,
                 finish_reward=100,
                 tower_target_size=4)

    params = HAMParams(q_value={},
                       env=env,
                       current_state=None,
                       eps=0.1,
                       gamma=0.9,
                       alpha=0.1,
                       string_prefix_of_machine=None,
                       accumulated_discount=1,
                       accumulated_rewards=0,
                       previous_machine_choice_state=None,
                       env_is_done=None,
                       logs={"reward": 0, "ep_rewards": []},
                       on_model_transition_id_function=lambda env_: 1 if env_.is_done() else 0,
                       )

    start = Start()
    choice_one = Choice()
    left = Action(action=env.get_actions_as_dict()["LEFT"])
    right = Action(action=env.get_actions_as_dict()["RIGHT"])
    up = Action(action=env.get_actions_as_dict()["UP"])
    down = Action(action=env.get_actions_as_dict()["DOWN"])
    on = Action(action=env.get_actions_as_dict()["ON"])
    off = Action(action=env.get_actions_as_dict()["OFF"])

    stop = Stop()
    simple_machine = AbstractMachine(
        transitions=(
            MachineRelation(left=start, right=choice_one),
            MachineRelation(left=choice_one, right=left),
            MachineRelation(left=choice_one, right=right),
            MachineRelation(left=choice_one, right=up),
            MachineRelation(left=choice_one, right=down),
            MachineRelation(left=choice_one, right=on),
            MachineRelation(left=choice_one, right=off),

            MachineRelation(left=left, right=stop, label=0),
            MachineRelation(left=right, right=stop, label=0),
            MachineRelation(left=up, right=stop, label=0),
            MachineRelation(left=down, right=stop, label=0),
            MachineRelation(left=on, right=stop, label=0),
            MachineRelation(left=off, right=stop, label=0),

            MachineRelation(left=left, right=stop, label=1),
            MachineRelation(left=right, right=stop, label=1),
            MachineRelation(left=up, right=stop, label=1),
            MachineRelation(left=down, right=stop, label=1),
            MachineRelation(left=on, right=stop, label=1),
            MachineRelation(left=off, right=stop, label=1),
        ),
    )

    sys.setrecursionlimit(100000)
    root = RootMachine(machine_to_invoke=LoopInvokerMachine(machine_to_invoke=simple_machine))
    num_episodes = 5000
    for i_episode in range(num_episodes):
        env.reset()
        root.run(params)
        if i_episode % 10 == 0:
            print("\r{root} episode {i_episode}/{num_episodes}.".format(**locals()), end="")
            sys.stdout.flush()
    plotting.plot_multi_test(smoothing_window=30,
                             xlabel="episode",
                             ylabel="smoothed rewards",
                             curve_to_draw=[params.logs["ep_rewards"]
                                            ],
                             labels=["ololo"]
                             )


if __name__ == "__main__":
    main()
