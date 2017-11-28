import numpy as np
import sys


def choice_update(info, choices, machine_state):
    last = info["last"]
    Q = info["Q"]
    env = info["env"]
    gamma = info["gamma"]
    r = info["r"]

    state = info["state"]
    if state not in Q[machine_state]:
        Q[machine_state][state] = np.zeros(len(choices))

    policy = np.ones(len(choices), dtype=float) * info["EPSILON"] / len(choices)
    best_action = np.argmax(Q[machine_state][state])
    policy[best_action] += (1.0 - info["EPSILON"])
    C = np.random.choice(np.arange(len(policy)), p=policy)
    if last is not None:
        last_machine_state, last_env_state, last_choice = last
        best_next_action = np.argmax(Q[machine_state][state])
        td_target = r + gamma * info["DIS_FACTOR"] * Q[machine_state][state][best_next_action]

        td_delta = td_target - Q[last_machine_state][last_env_state][last_choice]

        Q[last_machine_state][last_env_state][last_choice] += info["ALPHA"] * td_delta

    info["r"] = 0
    info["gamma"] = info["DIS_FACTOR"]
    info["last"] = (machine_state, state, C)

    return C


def apply_action(info, action):
    env = info["env"]
    next_state, reward, done, _ = env.step(action)
    info["r"] += reward
    info["gamma"] *= info["DIS_FACTOR"]
    info["state"] = next_state
    info["total_reward"] += reward
    info["actions_cnt"] += 1
    if done:
        info["done"] = True


def who_a_mi():
    return sys._getframe(1).f_code.co_name
