import numpy as np
import sys
from collections import defaultdict
from lib import plotting


class HAMParams:
    def __init__(self, Q, env, state, EPSILON, GAMMA, ALPHA, prefix_machine, logger):
        self.Q = Q
        self.env = env
        self.state = state
        self.EPSILON = EPSILON
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
        self.prefix_machine = prefix_machine

        self.accumulated_discount = 1.0
        self.accumulated_rewards = 0
        self.last = None
        self.done = False

        self.logger = logger


class HAM:
    @staticmethod
    def get_e_greedy(choices, info: HAMParams, machine_state):
        policy = np.ones(len(choices), dtype=float) * info.EPSILON / len(choices)
        best_action = np.argmax(info.Q[machine_state][info.state])
        policy[best_action] += (1.0 - info.EPSILON)
        return np.random.choice(np.arange(len(policy)), p=policy)

    @staticmethod
    def choice_update(info: HAMParams, choices, machine_state):

        state = info.state
        if state not in info.Q[machine_state]:
            info.Q[machine_state][state] = np.zeros(len(choices))

        if info.last is not None:
            last_machine_state, last_env_state, last_choice = info.last
            q = info.Q[last_machine_state][last_env_state][last_choice]
            V = info.Q[machine_state][state][np.argmax(info.Q[machine_state][state])]
            delta =info.ALPHA * (info.accumulated_rewards + info.accumulated_discount * V - q)
            q += delta
            # print(last_env_state, q)
            info.Q[last_machine_state][last_env_state][last_choice] = q
        c = HAM.get_e_greedy(choices=choices, info=info, machine_state=machine_state)

        info.accumulated_rewards = 0
        info.accumulated_discount = 1
        info.last = (machine_state, state, c)

        if info.logger is not None:
            info.logger.update(is_choice=True, loc=locals())
        return c

    @staticmethod
    def apply_action(info: HAMParams, action):
        if info.done:
            return

        info.state, reward, done, _ = info.env.step(action)

        info.accumulated_rewards += reward * info.accumulated_discount
        info.accumulated_discount *= info.GAMMA

        if done:
            info.done = True

        if info.logger is not None:
            info.logger.update(is_action=True, loc=locals())

    @staticmethod
    def call(info, machine):
        m = machine()
        old_prefix = info.prefix_machine
        info.prefix_machine += m.__class__.__name__
        m.start(info)
        info.prefix_machine = old_prefix

        if info.log is not None:
            info.log(is_call=True, loc=locals())

    @staticmethod
    def who_a_mi():
        return sys._getframe(1).f_code.co_name


def run_machine(info, machine):
    while not info.done:
        machine.start(info)
    return info


def ham_learning(env, num_episodes, GAMMA=0.9, ALPHA=0.1, EPSILON=0.1, machine=None, q=None, logger=None):
    if q is None:
        q = defaultdict(lambda: defaultdict(lambda: 0))
    assert (machine is not None)

    # Keeps track of useful statistics
    statistics = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following

    for i_episode in range(1, num_episodes + 1):
        if i_episode % 10 == 0:
            print("\r{machine} episode {i_episode}/{num_episodes}.".format(**locals()), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        info = HAMParams(Q=q,
                         env=env,
                         GAMMA=GAMMA,
                         state=env.reset(),
                         EPSILON=EPSILON,
                         ALPHA=ALPHA,
                         prefix_machine="",
                         logger=logger,
                         )

        info = run_machine(info, machine())
        # statistics.episode_lengths[i_episode - 1] = info.actions_cnt
        # statistics.episode_rewards[i_episode - 1] += info.total_reward
        # if episodes_info is not None:
        #     info.episodes_info.append(info.episode_info)
    return q, statistics
