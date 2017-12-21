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
