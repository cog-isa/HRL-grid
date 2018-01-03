from environments.arm_env import ArmEnv

import itertools
import numpy as np
import sys

from lib import plotting


def random_policy(env, num_episodes, ):
    stat = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.

        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        _ = env.reset()

        for t in itertools.count():
            # WE CAN PRINT ENVIRONMENT STATE
            # env.render()

            # Take a step
            action = np.random.choice(env.action_space.n, size=1)[0]
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stat.episode_rewards[i_episode] += reward
            stat.episode_lengths[i_episode] = t

            if done:
                break

            _ = next_state

    return stat


if __name__ == "__main__":
    a = ArmEnv()

    a.render()

    print("\n" * 100)

    c_env = ArmEnv()
    _, rew, is_done = (None, None, None)
    for i in range(100):
        print('\n' * 100)
        c_env.render()

        print(rew, is_done)

        print("0 LEFT")
        print("1 UP")
        print("2 RIGHT")
        print("3 DOWN")
        print("4 ON")
        print("5 OFF")

        while True:
            try:
                act = int(input())
                break
            except ValueError:
                pass
        _, reward, done, _ = c_env.step(act)
