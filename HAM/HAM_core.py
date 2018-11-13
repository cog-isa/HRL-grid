import random
import operator
import sys
from collections import defaultdict

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


class MachineGraph:
    def get_vertex_from_transitions(self):
        res = set(_.left for _ in self.transitions).union(set(_.right for _ in self.transitions))

        return res

    def get_vertex_mapping(self):
        res = defaultdict(lambda: [])
        for transition in self.transitions:
            res[transition.left].append(transition)
        return res
        # return {_: [i for i in self.transitions if i.left == _] for _ in self.vertices}

    def get_vertex_reverse_mapping(self):
        res = defaultdict(lambda: [])
        for transition in self.transitions:
            res[transition.right].append(transition)
        return res
        # return {_: [i for i in self.transitions if i.right == _] for _ in self.vertices}

    def get_action_vertex_label_mapping(self):
        return {_: {__.label: __ for __ in self.get_vertex_mapping()[_]} for _ in
                set([i.left for i in self.transitions if isinstance(i.left, Action)])}

    def get_special_vertices(self, special_vertex_class):
        return list(filter(lambda x: isinstance(x, special_vertex_class), self.vertices))

    def get_start(self):
        res = self.get_special_vertices(Start)
        assert (len(res) == 1)
        return res[0]

    def get_stop(self):
        res = self.get_special_vertices(Stop)
        assert (len(res) == 1)
        return res[0]

    def __init__(self, transitions, vertices=None):
        self.transitions = transitions
        self.vertices = vertices if vertices is not None else sorted(self.get_vertex_from_transitions(),
                                                                     key=lambda x: x.id)
        self.vertex_mapping = self.get_vertex_mapping()
        self.vertex_reverse_mapping = self.get_vertex_reverse_mapping()

        self.choice_relations = {__.left: {_.id: _ for _ in self.vertex_mapping[__.left]} for __ in transitions if
                                 isinstance(__.left, Choice)}
        self.action_vertex_label_mapping = {_: {__.label: __ for __ in self.vertex_mapping[_]} for _ in
                                            self.get_special_vertices(Action)}


class AbstractMachine:
    free_id = 1

    def __init__(self, graph: MachineGraph):
        self.graph = graph

        self.params = None
        self.get_on_model_transition_id = None

        self.previous_choice_state = None
        self.accumulated_discount = 1
        self.accumulated_rewards = 0
        self.V = defaultdict(lambda: None)
        # set unique id for AbstractMachine object
        self.id, AbstractMachine.free_id = AbstractMachine.free_id, AbstractMachine.free_id + 1

    def run(self, params: HAMParams):
        t = filter(lambda x: isinstance(x.left, Start), self.graph.transitions)
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

    def get_graph_to_draw(self, action_to_name_mapping=None, already_added_machines=None, no_edges_with_exit_f=None):
        if already_added_machines is None:
            already_added_machines = []
        graph = []
        for i in self.graph.transitions:
            if no_edges_with_exit_f and isinstance(i.left, Action) and i.label == 1:
                continue

            def get_str_with_special_for_actions(vertex):
                if isinstance(vertex, Action) and action_to_name_mapping is not None:
                    res = str(vertex)

                    for action_name, action_id in action_to_name_mapping.items():
                        res = res.replace("({action_id})".format(**locals()), "({action_name})".format(**locals()))
                    return res
                else:
                    return str(vertex)

            left_vertex = get_str_with_special_for_actions(i.left)
            right_vertex = get_str_with_special_for_actions(i.right)

            graph.append((left_vertex, right_vertex,
                          "f(E)=" + str(i.label) if i.label is not None and no_edges_with_exit_f is None else ""))

        for i in self.graph.transitions:

            if isinstance(i.right, Call):
                if i.right not in already_added_machines:
                    already_added_machines.append(i.right)
                    if i.right.machine_to_call is not None:
                        graph = graph + i.right.machine_to_call.get_graph_to_draw(
                            already_added_machines=already_added_machines,
                            action_to_name_mapping=action_to_name_mapping)
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
        super().__init__(graph=MachineGraph(transitions=transitions))


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
        super().__init__(graph=MachineGraph(transitions=transitions))


class RandomMachine(AbstractMachine):
    @staticmethod
    def create_random_vertex(env, machines_to_call=()):
        vertex_to_add_list = [Action(action=i) for i in sorted(env.get_actions_as_dict().values())]
        vertex_to_add_list += [Choice(), Choice()]
        vertex_to_add_list += [Call(machine_to_call=i) for i in machines_to_call]
        return random.choice(sorted(vertex_to_add_list, key=lambda x: x.id))

    @staticmethod
    def get_vertex_from_transitions(transitions):
        res = set(_.left for _ in transitions).union(set(_.right for _ in transitions))
        # remove auxiliary empty(None) vertex
        # if None in res:
        #     res.remove(None)
        return res

    @staticmethod
    def check_graph(graph):
        # TODO implement this

        # checking for duplicates
        for index_i, item_i in enumerate(graph.transitions):
            for index_j, item_j in enumerate(graph.transitions):
                if index_i >= index_j:
                    continue
                if item_i == item_j:
                    return False

        for vertex in graph.vertices:
            if isinstance(vertex, Call):
                if len(graph.vertex_mapping[vertex]) > 1:
                    return False

            elif isinstance(vertex, Action):
                # check for only single edge with definite label (on_model value)
                if len(graph.vertex_mapping[vertex]) > len(set(_.label for _ in graph.vertex_mapping[vertex])):
                    return False
            elif isinstance(vertex, Choice):
                pass
            elif isinstance(vertex, Stop):
                # no edges from Stop instance
                if len(graph.vertex_mapping[vertex]) > 0:
                    return False
            elif isinstance(vertex, Start):
                # no input edges for Start instance
                if len(graph.vertex_reverse_mapping[vertex]) > 0:
                    return False
                # single outer edge from Start instance
                if len(graph.vertex_mapping[vertex]) > 1:
                    return False
                    # if len(graph.vertex_mapping[vertex]) == 1 and isinstance(graph.vertex_mapping[vertex][0].right, Stop):
                    #     return False
            else:
                raise KeyError

        # p = AbstractMachine.get_action_vertex_label_mapping(transitions=transitions)
        return True

    @staticmethod
    def dfs_get_reachable_vertices(graph, vertex, reachable=None):
        if reachable is None:
            reachable = []
        if vertex in reachable:
            return reachable
        reachable.append(vertex)
        for relation in graph.vertex_mapping[vertex]:
            RandomMachine.dfs_get_reachable_vertices(graph=graph, vertex=relation.right, reachable=reachable)
        return reachable

    def get_new_possible_relation(self):
        # vertices = self.graph.vertices
        machine_relation_to_add = []

        # simple algorithm with complexity O(N^4) [one can done that with 0(N^2) complexity], but complexity is likely not an bottleneck in this case
        reachable_vertices = RandomMachine.dfs_get_reachable_vertices(graph=self.graph, vertex=self.graph.get_start())
        for index_i, left in enumerate(reachable_vertices):
            for index_j, right in enumerate(self.graph.vertices):
                new_machine_relation = MachineRelation(left=left, right=right, label=0) if isinstance(left,
                                                                                                      Action) else MachineRelation(
                    left=left, right=right)
                if RandomMachine.check_graph(
                        graph=MachineGraph(transitions=self.graph.transitions + [new_machine_relation],
                                           vertices=self.graph.vertices)):
                    machine_relation_to_add.append(new_machine_relation)

        assert (len(machine_relation_to_add) > 0)
        return random.choice(machine_relation_to_add)

    def __init__(self, graph=None):
        if graph is None:
            graph = MachineGraph(transitions=[], vertices=[Start(), Stop()])
        super().__init__(graph=graph)

    def with_new_vertex(self, env, machines_to_call=()):
        new_vertex = self.create_random_vertex(env=env, machines_to_call=machines_to_call)
        return RandomMachine(
            graph=MachineGraph(transitions=self.graph.transitions, vertices=self.graph.vertices + [new_vertex]))

    def with_new_relation(self):
        # res = MachineGraph(transitions=self.graph.transitions + [self.get_new_possible_relation()], vertices=self.graph.vertices)
        res = MachineGraph(transitions=self.graph.transitions, vertices=self.graph.vertices)
        stop = res.get_stop()
        # added to Action vertices link to the Stop with on_model env_done
        # TODO don't create links between unused vertex and Stop
        for vertex in res.get_special_vertices(Action):
            # print("::", res.graph.action_vertex_label_mapping[vertex])
            if not res.vertex_mapping[vertex] and not res.vertex_reverse_mapping[vertex]:
                continue
            if 1 not in res.action_vertex_label_mapping[vertex]:
                res.transitions.append(MachineRelation(left=vertex, right=stop, label=1))

        return RandomMachine(graph=MachineGraph(transitions=res.transitions, vertices=res.vertices))


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

        super().__init__(machine_to_invoke=LoopInvokerMachine(AbstractMachine(MachineGraph(transitions=transitions))))


class MachineVertex:
    free_id = 1

    def __init__(self):
        # set unique id for MachineVertex object
        self.id, MachineVertex.free_id = MachineVertex.free_id, MachineVertex.free_id + 1
        self.active = False

    def __str__(self):
        return "{self.__class__.__name__}{self.id}".format(**locals())

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def get_name(self):
        if isinstance(self, Start):
            return "Start"
        elif isinstance(self, Stop):
            return "Stop"
        elif isinstance(self, Choice):
            return "Choice"
        elif isinstance(self, Call):
            return "Call"
        elif isinstance(self, Action):
            return "Action", self.action
        else:
            raise TypeError

    def __lt__(self, other):
        def get_vertex_id(vertex):
            if isinstance(vertex, Start):
                return 0
            elif isinstance(vertex, Stop):
                return 1
            elif isinstance(vertex, Choice):
                return 2
            elif isinstance(vertex, Call):
                return 3
            elif isinstance(vertex, Action):
                return 4
            else:
                raise TypeError

        if isinstance(self, Action) and isinstance(other, Action):
            return self.action < other.action
        return get_vertex_id(self) < get_vertex_id(other)


class Start(MachineVertex):
    def run(self, own_machine: AbstractMachine):
        # return next vertex
        return own_machine.graph.vertex_mapping[self][0].right

    def __str__(self):
        return "{self.__class__.__name__}{self.id}".format(**locals()) + ("[A]" if self.active else "")


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
            own_machine.params.q_value[combined_state] = {_: 0 for _ in own_machine.graph.choice_relations[self].keys()}

        if own_machine.params.previous_machine_choice_state is not None:
            q = own_machine.params.q_value[own_machine.params.previous_machine_choice_state][
                own_machine.params.previous_machine_choice]
            v = own_machine.params.q_value[combined_state][
                self.get_e_greedy(own_machine.params.q_value[combined_state], eps=0)]
            delta = own_machine.params.alpha * (
                    own_machine.params.accumulated_rewards + own_machine.params.accumulated_discount * v - q)
            q += delta
            own_machine.params.q_value[own_machine.params.previous_machine_choice_state][
                own_machine.params.previous_machine_choice] = q

        action = self.get_e_greedy(own_machine.params.q_value[combined_state], eps=own_machine.params.eps)
        own_machine.params.previous_machine_choice_state = combined_state
        own_machine.params.previous_machine_choice = action

        own_machine.params.accumulated_rewards = 0
        own_machine.params.accumulated_discount = 1

        return own_machine.graph.choice_relations[self][action].right


class ChoiceSimple(Choice):
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
            own_machine.params.q_value[combined_state] = {_: 0 for _ in own_machine.graph.choice_relations[self].keys()}

        if own_machine.params.previous_machine_choice_state is not None:
            q = own_machine.params.q_value[own_machine.params.previous_machine_choice_state][
                own_machine.params.previous_machine_choice]


            v = own_machine.params.q_value[combined_state][
                self.get_e_greedy(own_machine.params.q_value[combined_state], eps=0)]
            own_machine.V[own_machine.params.env.get_current_state()] = (*own_machine.params.env.decode(own_machine.params.env.get_current_state()), v)
            delta = own_machine.params.alpha * (
                    own_machine.params.accumulated_rewards + own_machine.params.accumulated_discount * v - q)
            q += delta
            own_machine.params.q_value[own_machine.params.previous_machine_choice_state][
                own_machine.params.previous_machine_choice] = q

        action = self.get_e_greedy(own_machine.params.q_value[combined_state], eps=own_machine.params.eps)
        own_machine.params.previous_machine_choice_state = combined_state
        own_machine.params.previous_machine_choice = action

        own_machine.params.accumulated_rewards = 0
        own_machine.params.accumulated_discount = 1

        return own_machine.graph.choice_relations[self][action].right

class Call(MachineVertex):
    def __init__(self, machine_to_call: AbstractMachine):
        self.machine_to_call = machine_to_call
        super().__init__()

    def run(self, own_machine: AbstractMachine):
        self.machine_to_call.run(own_machine.params)

        # return next vertex
        return own_machine.graph.vertex_mapping[self][0].right

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
            # own_machine.params.env.render()
            own_machine.params.accumulated_rewards += reward * own_machine.params.accumulated_discount
            own_machine.params.accumulated_discount *= own_machine.params.gamma
            own_machine.params.eps *= 0.9999
            # if "gif" not in own_machine.params.logs:
            #     own_machine.params.logs["gif"] = [[]]
            # if done:
            #     own_machine.params.logs["gif"].append([])
            # own_machine.params.logs["gif"][-1].append(own_machine.params.env.get_grid())
        # return next vertex
        return own_machine.graph.action_vertex_label_mapping[self][own_machine.get_on_model_transition_id()].right


class ActionSimple(Action):
    def __str__(self):
        return super(ActionSimple, self).__str__() + "(" + str(self.action) + ")"

    def run(self, own_machine: AbstractMachine, reward):
        if self.action is not None:
            # state, reward, done, _ = own_machine.params.env.step(self.action)
            # own_machine.params.env.render()
            own_machine.params.accumulated_rewards += reward * own_machine.params.accumulated_discount
            own_machine.params.accumulated_discount *= own_machine.params.gamma
            own_machine.params.eps *= 0.9999
            # if "gif" not in own_machine.params.logs:
            #     own_machine.params.logs["gif"] = [[]]
            # if done:
            #     own_machine.params.logs["gif"].append([])
            # own_machine.params.logs["gif"][-1].append(own_machine.params.env.get_grid())
        # return next vertex
        return own_machine.graph.action_vertex_label_mapping[self][own_machine.get_on_model_transition_id()].right


class MachineRelation:
    free_id = 1

    def __init__(self, left, right, label=None):
        assert not (not isinstance(left,
                                   Action) and label is not None), "Action state vertex doesn't have specified label"
        assert not (isinstance(left, Action) and label is None), "Non action state vertex has specified label"

        self.left = left
        self.right = right
        self.label = label

        # set unique id for MachineRelation object
        self.id, MachineRelation.free_id = MachineRelation.free_id, MachineRelation.free_id + 1

    def __eq__(self, other):
        if self.id == other.id:
            return True
        if self.right == other.right and self.left == other.left and self.label == other.label:
            return True
        return False

    def __str__(self):
        return str(self.left) + " -> " + str(self.right)
