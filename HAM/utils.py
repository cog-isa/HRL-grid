import numpy as np
import sys
from collections import defaultdict
from lib import plotting


class HAM:
    @staticmethod
    def choice_update(info, choices, machine_state):
        last = info["last"]
        Q = info["Q"]
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

    @staticmethod
    def apply_action(info, action):
        env = info["env"]
        next_state, reward, done, _ = env.step(action)
        info["r"] += reward
        info["gamma"] *= info["DIS_FACTOR"]
        info["state"] = next_state
        info["total_reward"] += reward
        info["actions_cnt"] += 1

        #debug
        # if info["actions_cnt"] > 100000:
        #     env.render()
        if done:
            info["done"] = True

    @staticmethod
    def who_a_mi():
        return sys._getframe(1).f_code.co_name


def run_machine(info, machine):
    while not info["done"]:
        machine.start(info)
    return info


def ham_learning(env, num_episodes, discount_factor=0.9, alpha=0.1, epsilon=0.1, machine=None):
    q = defaultdict(lambda: defaultdict(lambda: 0))
    assert (machine is not None)

    # Keeps track of useful statistics
    statistics = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following

    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        info = {"Q": q,
                "env": env,
                "r": 0,
                "gamma": discount_factor,
                "last": None,
                "total_reward": 0,
                "actions_cnt": 0,
                "done": False,
                "state": state,
                "EPSILON": epsilon,
                "DIS_FACTOR": discount_factor,
                "ALPHA": alpha,
                "stats": statistics,
                }

        info = run_machine(info, machine())
        statistics.episode_lengths[i_episode] = info["actions_cnt"]
        statistics.episode_rewards[i_episode] += info["total_reward"]

    return q, statistics