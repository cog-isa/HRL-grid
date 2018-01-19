from environments.arm_env import ArmEnv


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
                 previous_machine_state,
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
        self.previous_machine_state = previous_machine_state
        self.env_is_done = env_is_done
        self.logs = logs
        self.on_model_transition_id_function = on_model_transition_id_function


class AbstractMachine:
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

        # set id's for choice points
        # TODO above

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
        choice = Choice(choice_id=0)
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
        print(self)
        pass


class Choice(MachineVertex):
    def __init__(self, choice_id):
        super(MachineVertex, self).__init__()
        self.id = choice_id

    def run(self, own_machine: AbstractMachine):
        print(self)

        # TODO implement choice vertex

        # calling next vertex
        return own_machine.vertex_mapping[self][0].right.run(own_machine)


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
        print(self)
        if self.action is not None:
            state, reward, done, _ = own_machine.params.env.step(self.action)

            own_machine.params.accumulated_rewards += reward * own_machine.params.accumulated_discount
            own_machine.params.accumulated_discount *= own_machine.params.gamma

        # calling next vertex
        return own_machine.action_vertex_label_mapping[self][own_machine.get_on_model_transition_id()].right.run(own_machine)


class MachineRelation:
    def __init__(self, left, right, label=None):
        assert not (not isinstance(left, Action) and label is not None), "Action state vertex doesn't have specified label"
        assert not (isinstance(left, Action) and label is None), "Non action state vertex has specified label"

        self.left = left
        self.right = right
        self.label = label

    def __str__(self):
        return str(self.left) + " -> " + str(self.right)


def main():
    env = ArmEnv(episode_max_length=10, size_x=5, size_y=4, cubes_cnt=5, action_minus_reward=-1, finish_reward=100,
                 tower_target_size=4)
    # TODO delete the code
    env.get_actions_as_dict()
    params = HAMParams(q_value={},
                       env=env,
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
                       on_model_transition_id_function=lambda env_: 1 if env_.is_done() else 0,
                       )

    start = Start()
    choice_one = Choice(choice_id=0)
    left = Action(action=env.get_actions_as_dict()["LEFT"])
    right = Action(action=env.get_actions_as_dict()["RIGHT"])
    up = Action(action=env.get_actions_as_dict()["UP"])
    down = Action(action=env.get_actions_as_dict()["DOWN"])
    on = Action(action=env.get_actions_as_dict()["ON"])
    off = Action(action=env.get_actions_as_dict()["OFF"])

    stop = Stop()
    simple_machine = AbstractMachine(transitions=(
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

    # simple_machine.run()

    # root = RootMachine(machine_to_invoke=RootMachine(machine_to_invoke=simple_machine))
    # root.run(params)
    root = RootMachine(machine_to_invoke=LoopInvokerMachine(machine_to_invoke=simple_machine))
    root.run(params)


if __name__ == "__main__":
    main()
