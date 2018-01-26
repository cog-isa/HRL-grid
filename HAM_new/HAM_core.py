import random
import operator
import sys

from environments.arm_env.arm_env import ArmEnv
from environments.env_core import CoreEnv
from utils import plotting


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

    @staticmethod
    def get_vertex_mapping(transitions):
        return {_: [i for i in transitions if i.left == _] for _ in set([i.left for i in transitions if i.left is not None])}

    @staticmethod
    def get_action_vertex_label_mapping(transitions):
        return {_: {__.label: __ for __ in AbstractMachine.get_vertex_mapping(transitions)[_]} for _ in
                set([i.left for i in transitions if isinstance(i.left, Action)])}

    def __init__(self, transitions):
        self.transitions = transitions

        self.vertex_mapping = self.get_vertex_mapping(transitions=transitions)

        self.action_vertex_label_mapping = self.get_action_vertex_label_mapping(transitions=transitions)

        # choice to relations_vertex id's mapping
        self.choice_relations = {__.left: {_.id: _ for _ in self.vertex_mapping[__.left]} for __ in transitions if isinstance(__.left, Choice)}

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
        # shortcut lambda for on_model function
        self.get_on_model_transition_id = lambda: self.params.on_model_transition_id_function(self.params.env)
        while not isinstance(current_vertex, Stop):
            current_vertex = current_vertex.run(self)

    def get_graph_to_draw(self, already_added_machines=set()):
        graph = []
        for i in self.transitions:
            graph.append((str(i.left), str(i.right), "f(E)=" + str(i.label) if i.label is not None else ""))
        for i in self.transitions:
            if isinstance(i.right, Call):
                if i.right not in already_added_machines:
                    already_added_machines.add(i.right)
                    graph = graph + i.right.machine_to_call.get_graph_to_draw(already_added_machines)
        return graph

    def __str__(self):
        return "{self.__class__.__name__}{self.id}".format(**locals())


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


class RandomMachine(AbstractMachine):
    @staticmethod
    def create_random_vertex(env, machines_to_call=()):
        vertex_to_add_list = [Action(action=i) for i in env.get_actions_as_dict().values()]
        vertex_to_add_list += [Choice()]
        vertex_to_add_list += [Call(machine_to_call=i) for i in machines_to_call]
        return random.choice(vertex_to_add_list)

    @staticmethod
    def get_vertex_from_transitions(transitions):
        res = set(_.left for _ in transitions).union(set(_.right for _ in transitions))
        # remove auxiliary empty(None) vertex
        if None in res:
            res.remove(None)
        return res

    @staticmethod
    def check_transitions(transitions):
        # TODO implement this
        vertex_mapping = AbstractMachine.get_vertex_mapping(transitions=transitions)

        # vertices = RandomMachine.get_vertex_from_transitions(transitions=transitions)

        for vertex in vertex_mapping:
            if isinstance(vertex, Call):
                if len(vertex_mapping[vertex]) != 1:
                    return False
            elif isinstance(vertex, Action):
                pass
            elif isinstance(vertex, Choice):
                pass
            elif isinstance(vertex, Stop):
                if len(vertex_mapping[vertex]) != 0:
                    return False
            elif isinstance(vertex, Start):
                if len(vertex_mapping[vertex]) != 1:
                    return False
            else:
                raise KeyError

        # p = AbstractMachine.get_action_vertex_label_mapping(transitions=transitions)
        return True

    @staticmethod
    def create_new_possible_relation(transitions):
        vertices = RandomMachine.get_vertex_from_transitions(transitions=transitions)
        machine_relation_to_add = []

        # simple algorithm with complexity O(N^4) [one can done that with 0(N^2) complexity], but complexity is likely not an bottleneck in this case
        for index_i, left in enumerate(vertices):
            for index_j, right in enumerate(vertices):
                if index_i > index_j:
                    continue
                new_machine_relation = MachineRelation(left=left, right=right, label=0) if isinstance(left, Action) else MachineRelation(left=left, right=right)
                for edge in transitions:
                    if isinstance(edge.left, Action) and isinstance(new_machine_relation.left, Action):
                        if edge.label == edge.label:
                            new_machine_relation = None
                            break
                    if edge.left == new_machine_relation.left and edge.right == new_machine_relation.right and edge.label == new_machine_relation.label:
                        new_machine_relation = None
                        break
                if new_machine_relation is None:
                    continue
                transitions.append(new_machine_relation)
                if RandomMachine.check_transitions(transitions=transitions):
                    machine_relation_to_add.append(new_machine_relation)
                transitions.pop()

        assert (len(machine_relation_to_add) > 0)
        return random.choice(machine_relation_to_add)

    def __init__(self, env: CoreEnv, machines_to_call=()):
        start = Start()
        stop = Stop()

        transitions = [MachineRelation(left=start, right=stop)]

        transitions.append(MachineRelation(left=None, right=self.create_random_vertex(env=env, machines_to_call=machines_to_call)))
        transitions.append(MachineRelation(left=None, right=self.create_random_vertex(env=env, machines_to_call=machines_to_call)))
        transitions.append(MachineRelation(left=None, right=self.create_random_vertex(env=env, machines_to_call=machines_to_call)))

        transitions.append(self.create_new_possible_relation(transitions=transitions))
        transitions.append(self.create_new_possible_relation(transitions=transitions))
        transitions.append(self.create_new_possible_relation(transitions=transitions))
        transitions.append(self.create_new_possible_relation(transitions=transitions))

        super().__init__(transitions=transitions)


class AutoBasicMachine(RootMachine):
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

        super().__init__(machine_to_invoke=LoopInvokerMachine(AbstractMachine(transitions=transitions)))


class MachineVertex:
    free_id = 1

    def __init__(self):
        # set unique id for MachineVertex object
        self.id, MachineVertex.free_id = MachineVertex.free_id, MachineVertex.free_id + 1

    def __str__(self):
        return "{self.__class__.__name__}{self.id}".format(**locals())

    def run(self, *args, **kwargs):
        raise NotImplementedError


class Start(MachineVertex):
    def run(self, own_machine: AbstractMachine):
        # return next vertex
        return own_machine.vertex_mapping[self][0].right

    def __str__(self):
        return "{self.__class__.__name__}{self.id}".format(**locals())


class Stop(MachineVertex):
    def run(self, own_machine: AbstractMachine):
        pass


class Choice(MachineVertex):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_e_greedy(q_choices: dict, eps: float):
        if random.random() < eps:
            return random.choice(list(q_choices.keys()))
        else:
            return max(q_choices.items(), key=operator.itemgetter(1))[0]

    def run(self, own_machine: AbstractMachine):
        combined_state = own_machine.id, self.id, own_machine.params.env.get_current_state()

        if combined_state not in own_machine.params.q_value:
            own_machine.params.q_value[combined_state] = {_: 0 for _ in own_machine.choice_relations[self].keys()}

        if own_machine.params.previous_machine_choice_state is not None:
            q = own_machine.params.q_value[own_machine.params.previous_machine_choice_state][own_machine.params.previous_machine_choice]
            v = own_machine.params.q_value[combined_state][self.get_e_greedy(own_machine.params.q_value[combined_state], eps=0)]
            delta = own_machine.params.alpha * (own_machine.params.accumulated_rewards + own_machine.params.accumulated_discount * v - q)
            q += delta
            own_machine.params.q_value[own_machine.params.previous_machine_choice_state][own_machine.params.previous_machine_choice] = q

        action = self.get_e_greedy(own_machine.params.q_value[combined_state], eps=own_machine.params.eps)
        own_machine.params.previous_machine_choice_state = combined_state
        own_machine.params.previous_machine_choice = action

        own_machine.params.accumulated_rewards = 0
        own_machine.params.accumulated_discount = 1

        return own_machine.choice_relations[self][action].right


class Call(MachineVertex):
    def __init__(self, machine_to_call: AbstractMachine):
        self.machine_to_call = machine_to_call
        super().__init__()

    def run(self, own_machine: AbstractMachine):
        self.machine_to_call.run(own_machine.params)

        # return next vertex
        return own_machine.vertex_mapping[self][0].right

    def __str__(self):
        return super().__str__() + "[{self.machine_to_call}]".format(**locals())


class Action(MachineVertex):
    def __init__(self, action=None):
        self.action = action
        super().__init__()

    def __str__(self):
        return super(Action, self).__str__() + "(" + str(self.action) + ")"

    def run(self, own_machine: AbstractMachine):
        if self.action is not None:
            state, reward, done, _ = own_machine.params.env.step(self.action)
            own_machine.params.logs["reward"] += reward
            if done:
                own_machine.params.logs["ep_rewards"].append(own_machine.params.logs["reward"])
                own_machine.params.logs["reward"] = 0

            own_machine.params.accumulated_rewards += reward * own_machine.params.accumulated_discount
            own_machine.params.accumulated_discount *= own_machine.params.gamma

        # return next vertex
        return own_machine.action_vertex_label_mapping[self][own_machine.get_on_model_transition_id()].right


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
                 size_y=3,
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

    root = RootMachine(machine_to_invoke=LoopInvokerMachine(machine_to_invoke=simple_machine))
    num_episodes = 1500
    for i_episode in range(num_episodes):
        env.reset()
        root.run(params)
        if i_episode % 10 == 0:
            print("\r{root} episode {i_episode}/{num_episodes}.".format(**locals()), end="")
            sys.stdout.flush()
    plotting.plot_multi_test(smoothing_window=30,
                             x_label="episode",
                             y_label="smoothed rewards",
                             curve_to_draw=[params.logs["ep_rewards"]
                                            ],
                             labels=["HAM_basic"]
                             )


if __name__ == "__main__":
    main()
