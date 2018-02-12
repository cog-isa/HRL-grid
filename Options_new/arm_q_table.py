import itertools
import sys
import numpy as np
from environments.arm_env.arm_env import ArmEnv
from utils import plotting


class Arm2(ArmEnv):

    def get_tower_height(self):
        h = 0
        for j in range(self._grid.shape[1]):
            t = 0
            for i in np.arange(self._grid.shape[0]-1, 0, -1):
                if self._grid[i, j] == 1 and self._grid[i-1, j] == 0 and (i+1 == self._grid.shape[0] or self._grid[i+1, j] == 1):
                    t = self._grid.shape[0] - i
                    break
            if t > h:
                h = t
        return h

def q_learning(env, num_episodes, eps=0.5, alpha=0.5, gamma=1.0):
    to_plot = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    q_table = {}

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.

        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
            eps = eps - 0.1 * eps

        # Reset the environment and pick the first action
        env.reset()
        state = env.get_current_state()
        if state not in q_table:
            q_table[state] = np.zeros(shape=env.action_space.n)

        for t in itertools.count():
            # WE CAN PRINT ENVIRONMENT STATE
            # env.render()

            # Take a step
            if np.random.rand(1) < eps:  # choose random action
                action = np.random.choice(env.action_space.n, size=1)[0]
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _ = env.step(action)
            if next_state not in q_table:
                q_table[next_state] = np.zeros(shape=env.action_space.n)
            q_table[state][action] = (1 - alpha) * q_table[state][action] + \
                                     alpha * (reward + gamma * np.max(q_table[next_state]))

            # Update statistics
            to_plot.episode_rewards[i_episode] += reward
            to_plot.episode_lengths[i_episode] = t

            if done:
                break

            state = next_state

    return to_plot, q_table



def test_policy(env, q_table):
    moves_d = {0: 'LEFT', 1: "UP", 2: "RIGHT", 3: "DOWN", 4: "ON", 5: "OFF", 6: "OPTION"}
    env.reset()
    state = env.get_current_state()
    S_r = 0
    S_t = 0
    print("\n Start of the episode")
    env.render()
    print("Tower height:", env.get_tower_height())

    for t in itertools.count():
        # WE CAN PRINT ENVIRONMENT STATE

        # Take a step
        action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)
        print("Next move:", moves_d[action])
        print("\n Got reward ", reward)

        # Update statistics
        S_r += reward
        S_t = t

        if done:
            env.render()
            print("Tower height:", env.get_tower_height())
            print("\nEnd of the episode")
            break

        state = next_state
        env.render()
        print("Tower height:", env.get_tower_height())
    return S_r, S_t


def main():
    env = Arm2(episode_max_length=50,
                 size_x=5,
                 size_y=3,
                 cubes_cnt=4,
                 action_minus_reward=-1,
                 finish_reward=100,
                 tower_target_size=4)
    stats, q_table = q_learning(env, 4000)
    plotting.plot_episode_stats(stats)

    S, t = test_policy(env, q_table)
    print("Time: ", t, "Reward: ", S)


if __name__ == '__main__':
    main()
