def start(info):
    return choice(info)


def choice(info):
    choices = [action_right, action_left, action_down, action_up]
    name = __name__ + ":" + info["me"]()
    choices[info["choice_update"](info, choices, name)](info)


def action_right(info):
    action = 0
    info["apply_action"](info, action)
    stop(info)


def action_down(info):
    action = 1
    info["apply_action"](info, action)
    stop(info)


def action_up(info):
    action = 2
    info["apply_action"](info, action)
    stop(info)


def action_left(info):
    action = 3
    info["apply_action"](info, action)
    stop(info)


def stop(info):
    pass
