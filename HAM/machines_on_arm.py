from HAM.utils import HAM

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3
ON = 4
OFF = 5


class BasicMachineArm(HAM):
    def start(self, info):
        return self.choice(info)

    def choice(self, info):
        choices = [self.action_up, self.action_right, self.action_down, self.action_left, self.action_on, self.action_off]
        name = info.prefix_machine + __name__ + ":" + self.who_a_mi()
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

    def action_on(self, info):
        action = ON
        self.apply_action(info, action)
        self.stop(info)

    def action_off(self, info):
        action = OFF
        self.apply_action(info, action)
        self.stop(info)

    def stop(self, info):
        pass


class SmartMachine(BasicMachineArm):
    def start(self, info):
        return self.choice(info)

    def choice(self, info):
        choices = [self.action_right, self.action_on, self.action_off, self.action_left, self.action_off, self.call_smart]
        name = info.prefix_machine + __name__ + ":" + self.who_a_mi()
        choices[self.choice_update(info, choices, name)](info)

    def call_smart(self, info):
        self.call(info, Smart)
        self.choice(info)


class Smart(HAM):
    def start(self, info):
        return self.action_on(info)

    def action_on(self, info):
        action = ON
        self.apply_action(info, action)
        self.action_down_1(info)

    def action_down_1(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.action_down_2(info)

    def action_down_2(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.action_down_3(info)

    def action_down_3(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.action_up_1(info)

    def action_up_1(self, info):
        action = UP
        self.apply_action(info, action)
        self.action_up_2(info)

    def action_up_2(self, info):
        action = UP
        self.apply_action(info, action)
        self.action_up_3(info)

    def action_up_3(self, info):
        action = UP
        self.apply_action(info, action)
        # end


class SmartMachineTest(BasicMachineArm):
    def start(self, info):
        return self.choice(info)

    def choice(self, info):
        choices = [self.action_right, self.action_on, self.action_off, self.action_left, self.action_off, self.action_on]
        name = info.prefix_machine + __name__ + ":" + self.__class__.__name__ + ":" + self.who_a_mi()
        # print(name)
        choices[self.choice_update(info, choices, name)](info)

    def action_on(self, info):
        action = ON
        self.apply_action(info, action)
        self.action_down_1(info)

    def action_down_1(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.action_down_2(info)

    def action_down_2(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.action_down_3(info)

    def action_down_3(self, info):
        action = DOWN
        self.apply_action(info, action)
        self.action_up_1(info)

    def action_up_1(self, info):
        action = UP
        self.apply_action(info, action)
        self.action_up_2(info)

    def action_up_2(self, info):
        action = UP
        self.apply_action(info, action)
        self.action_up_3(info)

    def action_up_3(self, info):
        action = UP
        self.apply_action(info, action)
