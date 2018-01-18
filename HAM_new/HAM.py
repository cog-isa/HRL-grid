from collections import namedtuple

from environments.arm_env import ArmEnv

HAM_params = namedtuple("HAM_params", ['q_value',
                                       'env',
                                       'current_state',
                                       'eps',
                                       'gamma',
                                       'alpha',
                                       'string_prefix_of_machine',
                                       'accumulated_discount',
                                       'accumulated_rewards',
                                       'previous_machine_state',
                                       'env_is_done',
                                       'logs',
                                       'on_model_transition_id_function'])


class AbstractMachine:
    def __init__(self, transitions):
        self.transitions = transitions

        self.vertex_mapping = {_: [i for i in self.transitions if i.left == _] for _ in set([i.left for i in transitions])}

        self.action_vertex_label_mapping = {_: {__.label: __ for __ in self.vertex_mapping[_]} for _ in
                                            set([i.left for i in transitions if isinstance(i.left, Action)])}

        self.params = None
        self.get_on_model_transition_id = None
        # self.params = params
        # self.get_on_model_transition_id = lambda params: params.on_model_transition_id_function(params.env)

        self.previous_choice_state = None
        self.accumulated_discount = 1
        self.accumulated_rewards = 0

    def run(self, params):
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
        self.get_on_model_transition_id = lambda: self.params.on_model_transition_id_function(self.params.env)
        current_vertex.run(self)
        self.params = None
        self.get_on_model_transition_id = None


class RootMachine(AbstractMachine):
    def __init__(self, machine_to_invoke):
        start = Start()
        call = Call(machine_to_invoke)
        choice = Choice()
        stop = Stop()
        transitions = (
            Relation(left=start, right=call),
            Relation(left=call, right=choice),
            Relation(left=choice, right=stop)
        )
        super(RootMachine, self).__init__(transitions=transitions)


class LoopInvokerMachine(AbstractMachine):
    pass


class MachineVertex:
    def __str__(self):
        return self.__class__.__name__

    def run(self, *args, **kwargs):
        raise NotImplementedError


class Start(MachineVertex):
    def run(self, own_machine: AbstractMachine):
        own_machine.vertex_mapping[self][0].right.run(own_machine)


class Stop(MachineVertex):
    def run(self, own_machine: AbstractMachine):
        print(self)
        pass


class Choice(MachineVertex):
    def run(self, own_machine: AbstractMachine):
        # TODO do smth
        print(self)
        own_machine.vertex_mapping[self][0].right.run(own_machine)


class Call(MachineVertex):
    def __init__(self, machine_to_call: AbstractMachine):
        super(MachineVertex, self).__init__()
        self.machine_to_call = machine_to_call

    def run(self, own_machine: AbstractMachine):
        # TODO do smth
        print(self)
        self.machine_to_call.run(own_machine.params)
        own_machine.vertex_mapping[self][0].right.run(own_machine)


class Action(MachineVertex):
    def __init__(self, action=None):
        self.action = action

    def __str__(self):
        return super(Action, self).__str__() + "(" + str(self.action) + ")"

    def run(self, own_machine):
        # TODO do smth
        print(self)
        own_machine.action_vertex_label_mapping[self][own_machine.get_on_model_transition_id()].right.run(own_machine)
        if self.action is not None:
            # apply_action(self.action)
            pass


class Relation:
    def __init__(self, left, right, label=None):
        assert not (not isinstance(left, Action) and label is not None), "Action state vertex doesn't have specified label"
        assert not (isinstance(left, Action) and label is None), "Non action state vertex has specified label"

        self.left = left
        self.right = right
        self.label = label

    def __str__(self):
        return str(self.left) + " -> " + str(self.right)


def main():
    params = HAM_params(q_value={},
                        env=ArmEnv(episode_max_length=100, size_x=5, size_y=4, cubes_cnt=5, action_minus_reward=-1, finish_reward=100,
                                   tower_target_size=4),
                        current_state=None,
                        eps=0.1,
                        gamma=0.9,
                        alpha=0.1,
                        string_prefix_of_machine=None,
                        accumulated_discount=1,
                        accumulated_rewards=0,
                        previous_machine_state=None,
                        env_is_done=None,
                        logs=None,
                        on_model_transition_id_function=lambda env: 1 if env.is_done() else 0,
                        )

    start = Start()
    choice_one = Choice()
    left = Action()
    right = Action()
    up = Action()
    down = Action()
    on = Action()
    off = Action()

    stop = Stop()
    simple_machine = AbstractMachine(transitions=(
        Relation(left=start, right=choice_one),
        Relation(left=choice_one, right=left),
        Relation(left=choice_one, right=right),
        Relation(left=choice_one, right=up),
        Relation(left=choice_one, right=down),
        Relation(left=choice_one, right=on),
        Relation(left=choice_one, right=off),

        Relation(left=left, right=stop, label=0),
        Relation(left=right, right=stop, label=0),
        Relation(left=up, right=stop, label=0),
        Relation(left=down, right=stop, label=0),
        Relation(left=on, right=stop, label=0),
        Relation(left=off, right=stop, label=0),

        Relation(left=left, right=stop, label=1),
        Relation(left=right, right=stop, label=1),
        Relation(left=up, right=stop, label=1),
        Relation(left=down, right=stop, label=1),
        Relation(left=on, right=stop, label=1),
        Relation(left=off, right=stop, label=1),
    ),

        # params=params,
    )

    # simple_machine.run()

    root = RootMachine(machine_to_invoke=RootMachine(machine_to_invoke=simple_machine))
    root.run(params)


if __name__ == "__main__":
    main()

    # TODO should begin from realisation of LoopInvokerMachine
