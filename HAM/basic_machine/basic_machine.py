UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


def start(info):
    return choice(info)


def choice(info):
    choices = [action_right, action_left, action_down, action_up]
    name = __name__ + ":" + info["me"]()
    choices[info["choice_update"](info, choices, name)](info)


def action_right(info):
    action = RIGHT
    info["apply_action"](info, action)
    stop(info)


def action_down(info):
    action = DOWN
    info["apply_action"](info, action)
    stop(info)


def action_up(info):
    action = UP
    info["apply_action"](info, action)
    stop(info)


def action_left(info):
    action = LEFT
    info["apply_action"](info, action)
    stop(info)


def stop(info):
    pass
