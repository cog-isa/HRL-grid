import sys
import numpy as np
from HAM.utils import HAM

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class BasicMachine(HAM):
    def start(self, info):
        return self.choice(info)

    def choice(self, info):
        choices = [self.action_up, self.action_right, self.action_down, self.action_left]
        name = __name__ + ":" + self.who_a_mi()
        choices[self.choice_update(info, choices, name)](info)

    def action_right(self, info):
        action = RIGHT
        self.apply_action(info, action)
        self.stop(info)

    def action_down(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.stop(info)

    def action_up(self, info):
        action = UP
        self.apply_action(info, action)
        self.stop(info)

    def action_left(self, info):
        action = LEFT
        self.apply_action(info, action)
        self.stop(info)

    def stop(self, info):
        pass


class l1_machine(HAM):
    def start(self, info):
        return self.choice(info)

    def choice(self, info):
        choices = [self.action_up, self.action_right, self.action_down, self.action_left, self.a1]
        name = __name__ + ":" + self.who_a_mi()
        choices[self.choice_update(info, choices, name)](info)

    def action_right(self, info):
        action = RIGHT
        self.apply_action(info, action)
        self.stop(info)

    def action_down(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.stop(info)

    def action_up(self, info):
        action = UP
        self.apply_action(info, action)
        self.stop(info)

    def action_left(self, info):
        action = LEFT
        self.apply_action(info, action)
        self.stop(info)

    def a1(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.a2(info)

    def a2(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.a3(info)

    def a3(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.a4(info)

    def a4(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.a5(info)

    def a5(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.stop(info)

    def stop(self, info):
        pass


class l2_machine(HAM):
    def start(self, info):
        return self.choice(info)

    def choice(self, info):
        choices = [BasicMachine().start, Left5().start]
        name = __name__ + ":" + self.who_a_mi()
        choices[self.choice_update(info, choices, name)](info)

    def stop(self, info):
        pass


class Left5(HAM):
    def start(self, info):
        return self.l1(info)

    def l1(self, info):
        action = LEFT
        self.apply_action(info, action)
        self.l2(info)

    def l2(self, info):
        action = LEFT
        self.apply_action(info, action)
        self.l3(info)

    def l3(self, info):
        action = LEFT
        self.apply_action(info, action)
        self.l4(info)

    def l4(self, info):
        action = LEFT
        self.apply_action(info, action)
        self.l5(info)

    def l5(self, info):
        action = LEFT
        self.apply_action(info, action)
        self.stop(info)

    def stop(self, info):
        pass
