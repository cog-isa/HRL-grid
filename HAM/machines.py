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
        name = info["prefix_machine"] + __name__ + ":" + self.who_a_mi()
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


class L1Machine(HAM):
    def start(self, info):
        return self.choice(info)

    def choice(self, info):
        choices = [self.action_up, self.action_right, self.action_down, self.action_left, self.a1]
        name = info["prefix_machine"] + __name__ + ":" + self.who_a_mi()
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


class L2Move5(HAM):
    def start(self, info):
        return self.choice(info)

    def choice(self, info):
        choices = [self.call_basic, self.call_left, self.call_up, self.call_right, self.call_down]
        name = info["prefix_machine"] + __name__ + ":" + self.who_a_mi()
        choices[self.choice_update(info, choices, name)](info)

    def call_basic(self, info):
        self.call(info, BasicMachine)

    def call_left(self, info):
        self.call(info, Left5)

    def call_right(self, info):
        self.call(info, Right5)

    def call_up(self, info):
        self.call(info, Up5)

    def call_down(self, info):
        self.call(info, Down5)

    def stop(self, info):
        pass


class L2Move3(HAM):
    def start(self, info):
        return self.choice(info)

    def choice(self, info):
        choices = [self.call_basic, self.call_up3, self.call_down3, self.call_left3, self.call_right3]
        name = info["prefix_machine"] + __name__ + ":" + self.who_a_mi()
        choices[self.choice_update(info, choices, name)](info)

    def call_basic(self, info):
        self.call(info, BasicMachine)

    def call_left3(self, info):
        self.call(info, Left3)

    def call_right3(self, info):
        self.call(info, Right3)

    def call_up3(self, info):
        self.call(info, Up3)

    def call_down3(self, info):
        self.call(info, Down3)

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


class Right5(HAM):
    def start(self, info):
        return self.l1(info)

    def l1(self, info):
        action = RIGHT
        self.apply_action(info, action)
        self.l2(info)

    def l2(self, info):
        action = RIGHT
        self.apply_action(info, action)
        self.l3(info)

    def l3(self, info):
        action = RIGHT
        self.apply_action(info, action)
        self.l4(info)

    def l4(self, info):
        action = RIGHT
        self.apply_action(info, action)
        self.l5(info)

    def l5(self, info):
        action = RIGHT
        self.apply_action(info, action)
        self.stop(info)

    def stop(self, info):
        pass


class Up5(HAM):
    def start(self, info):
        return self.l1(info)

    def l1(self, info):
        action = UP
        self.apply_action(info, action)
        self.l2(info)

    def l2(self, info):
        action = UP
        self.apply_action(info, action)
        self.l3(info)

    def l3(self, info):
        action = UP
        self.apply_action(info, action)
        self.l4(info)

    def l4(self, info):
        action = UP
        self.apply_action(info, action)
        self.l5(info)

    def l5(self, info):
        action = UP
        self.apply_action(info, action)
        self.stop(info)

    def stop(self, info):
        pass


class Down5(HAM):
    def start(self, info):
        return self.l1(info)

    def l1(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.l2(info)

    def l2(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.l3(info)

    def l3(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.l4(info)

    def l4(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.l5(info)

    def l5(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.stop(info)

    def stop(self, info):
        pass


class Left3(HAM):
    def start(self, info):
        return self.l3(info)

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


class Right3(HAM):
    def start(self, info):
        return self.l3(info)

    def l3(self, info):
        action = RIGHT
        self.apply_action(info, action)
        self.l4(info)

    def l4(self, info):
        action = RIGHT
        self.apply_action(info, action)
        self.l5(info)

    def l5(self, info):
        action = RIGHT
        self.apply_action(info, action)
        self.stop(info)

    def stop(self, info):
        pass


class Up3(HAM):
    def start(self, info):
        return self.l3(info)

    def l3(self, info):
        action = UP
        self.apply_action(info, action)
        self.l4(info)

    def l4(self, info):
        action = UP
        self.apply_action(info, action)
        self.l5(info)

    def l5(self, info):
        action = UP
        self.apply_action(info, action)
        self.stop(info)

    def stop(self, info):
        pass


class Down3(HAM):
    def start(self, info):
        return self.l3(info)

    def l3(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.l4(info)

    def l4(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.l5(info)

    def l5(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.stop(info)

    def stop(self, info):
        pass


class L2Interesting(HAM):
    def start(self, info):
        return self.choice(info)

    def choice(self, info):
        choices = [self.call_left, self.call_up, self.call_right, self.call_down]
        name = info["prefix_machine"] + __name__ + ":" + self.who_a_mi()
        choices[self.choice_update(info, choices, name)](info)

    def call_left(self, info):
        self.call(info, Left5)
        self.call_basic(info)

    def call_right(self, info):
        self.call(info, Right5)
        self.call_basic(info)

    def call_down(self, info):
        self.call(info, Down5)
        self.call_basic(info)

    def call_up(self, info):
        self.call(info, Up5)
        self.call_basic(info)

    def call_basic(self, info):
        self.call(info, BasicMachine)
        self.stop(info)

    def stop(self, info):
        pass
