import numpy as np
import sys
from collections import defaultdict
from lib import plotting

from collections import namedtuple

namedtuple("HAM_params", ['q_value',
                          'environment',
                          'current_state',
                          'eps',
                          'gamma',
                          'alpha',
                          'string_prefix_of_machine',
                          'accumulated_discount',
                          'accumulated_rewards',
                          'previous_machine_state',
                          'env_is_done',
                          'logs'])


class AbstractMachine:
    def __init__(self, transitions):
        self.transitions = transitions

    def run(self):
        t = filter(lambda x: isinstance(x.left_vertex, Start), self.transitions)
        start = t.__next__()
        try:
            assert (t.__next__(), "More than one start vertex in graph")
        except StopIteration:
            pass


class Vertex:
    pass


class Start(Vertex):
    pass


class Stop:
    pass


class Choice(Vertex):
    pass


class Call(Vertex):
    pass


class Action(Vertex):
    pass


class Relation:
    def __init__(self, left_vertex, right_vertex):
        self.left_vertex = left_vertex
        self.right_vertex = right_vertex


def main():
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
        Relation(start, choice_one),
        Relation(choice_one, left),
        Relation(choice_one, right),
        Relation(choice_one, up),
        Relation(choice_one, down),
        Relation(choice_one, on),
        Relation(choice_one, off),

        Relation(left, stop),
        Relation(right, stop),
        Relation(up, stop),
        Relation(down, stop),
        Relation(on, stop),
        Relation(off, stop),
    ))

    simple_machine.run()


if __name__ == "__main__":
    main()
    # smart_machine = AbstractMachine(transitions=[
    # start -> choice_one
    # choice_one -> right, on, off, left, call_smart
    # right, on-model=0 -> choice_one
    # on, on-model=0 -> choice_one
    # off, on-model=0 -> choice_one
    # left, on-model=0 -> choice_one
    # call_smart -> choice_one
    #
    # ])
