import numpy as np
from Options_new.maze_options import MazeWorldTrain, place_finish, prepare_train_maze, Option
import sys
from environments.maze_world_env import MazeWorld, MazeWorldEpisodeLength
from lib import plotting
import itertools

from environments.grid_maze_generator import place_start_finish, generate_maze, generate_pattern, prepare_maze

#create environment
ids = [2, 32, 64, 256, 1024]
blocks = [generate_pattern(i) for i in ids]
maze, pattern_no_to_id = generate_maze(blocks=blocks, size_x=8, size_y=8, options=True)
m = place_start_finish(prepare_maze(maze))
env = MazeWorldEpisodeLength(maze=m, episode_max_length=700, finish_reward=500)

env.render()

# for each type of pattern create 4 train environments
train_envs = np.array([[MazeWorldTrain(place_finish(prepare_train_maze(generate_pattern(id)), direction=a))
               for a in range(4)] for id in ids])

# for each type of pattern we have 4 options (even if some ways are blocked)
maze_options = np.array([[Option(train_env_).learning_option() for train_env_ in train_env] for train_env in train_envs])


def q_learning_on_options(env, pattern_no_to_id, options, num_episodes, eps=0.7, alpha=0.1, gamma=0.99):
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    n_actions = env.action_space.n
    n_options = len(options[0])
    n_states = env.observation_space.n

    # initialize q-function
    q_table = np.zeros(shape=(n_states, n_actions + n_options))

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.

        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
            eps = eps - 0.1 * eps

        # Reset the environment
        state = env.reset()

        for t in itertools.count():
            # WE CAN PRINT ENVIRONMENT STATE
            #if i_episode == num_episodes - 1:
            #    print("\n")
            #    print(state)
            #    env.render()

            # Take an action
            if np.random.rand(1) < eps:  # choose random option
                action0 = np.random.choice(np.arange(n_actions + n_options), size=1)[0]
            else:
                action0 = np.argmax(q_table[state, :])

            #if i_episode == num_episodes - 1: print(action0)

            # if option is chosen
            if action0 < 4:
                pattern_no = env.state_to_pattern_no[state]
                pattern_id = pattern_no_to_id[pattern_no]

                #if i_episode == num_episodes - 1: print(pattern_no, pattern_id)

                opt = options[pattern_id, action0]
                opt_rew = 0
                opt_t = 0
                opt_state = state

                while opt_t < 10:

                    opt_state_no = env.state_to_state_no[opt_state]
                    index = options[pattern_id, 0].state_no_to_index[opt_state_no]

                    action = np.argmax(opt.q_table[index, :])
                    opt_state, reward, done, _ = env.step(action)

                    opt_rew += reward
                    opt_t += 1

                    if done or (env.state_to_pattern_no[opt_state] != pattern_no):
                        break

                next_state = opt_state

                q_table[state, action0] = (1 - alpha) * q_table[state, action0] + alpha * (
                opt_rew + gamma**opt_t * np.max(q_table[next_state, :]))

                # Update statistics
                stats.episode_rewards[i_episode] += opt_rew
                stats.episode_lengths[i_episode] = t

                if done:
                    break

                state = next_state

            else:
                next_state, reward, done, _ = env.step(action0-4)
                q_table[state, action0] = (1 - alpha) * q_table[state, action0] + alpha * (
                    reward + gamma * np.max(q_table[next_state, :]))

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                if done:
                    break

                state = next_state

    return stats, q_table


#stats_opt, q_table_opt = q_learning_on_options(env, pattern_no_to_id, maze_options, 8000)
#plotting.plot_episode_stats(stats_opt)

def test_policy(env, q_table, options):
    state = env.reset()
    s_r = 0
    s_t = 0
    n_actions = env.action_space.n
    n_options = len(options[0])
    n_states = env.observation_space.n

    for t in itertools.count():
        # WE CAN PRINT ENVIRONMENT STATE
        env.render()
        print(state)

        # Take a step
        action = np.argmax(q_table[state, :])
        print(action)

        if action < 4:
            pattern_no = env.state_to_pattern_no[state]
            pattern_id = pattern_no_to_id[pattern_no]

            opt = options[pattern_id, action]
            # execute the option's policy
            opt_rew = 0
            opt_t = 0
            opt_state = state

            while opt_t < 10:

                opt_state_no = env.state_to_state_no[opt_state]
                index = options[pattern_id, 0].state_no_to_index[opt_state_no]

                action = np.argmax(opt.q_table[index, :])
                opt_state, reward, done, _ = env.step(action)

                opt_rew += reward
                opt_t += 1

                if done or (env.state_to_pattern_no[opt_state] != pattern_no):
                    break

            next_state = opt_state
            reward = opt_rew

            s_r += reward
            s_t = t

            if done:
                break

            state = next_state
        else:
            next_state, reward, done, _ = env.step(action-4)

            s_r += reward
            s_t = t

            if done:
                break

        state = next_state
    return s_r, s_t

#print("\n Testing policy")
#s, t = test_policy(env, q_table_opt, maze_options)
#print(s, t)


def q_learning(env, num_episodes, eps=0.7, alpha=0.1, gamma=0.99):
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # initialize q-function
    q_table = np.zeros(shape=(env.observation_space.n, env.action_space.n))

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.

        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
            eps = eps - 0.1 * eps

        # Reset the environment and pick the first action
        state = env.reset()

        for t in itertools.count():
            # WE CAN PRINT ENVIRONMENT STATE
            # env.render()

            # Take a step
            if np.random.rand(1) < eps:  # choose random action
                action = np.random.choice(env.action_space.n, size=1)[0]
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, done, _ = env.step(action)
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state, :]))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break

            state = next_state

    return stats, q_table

a = 5000
stats_opt, q_table_opt = q_learning_on_options(env, pattern_no_to_id, maze_options, a)
stats, q_table = q_learning(env, a)

plotting.plot_multi_test(smoothing_window=30,
                             xlabel="episode",
                             ylabel="smoothed rewards",
                             curve_to_draw=[stats_opt.episode_rewards,
                                            stats.episode_rewards],
                             labels=["options", "q-learning"]
                             )

#plotting.plot_multi_test(smoothing_window=30,
#                             xlabel="episode",
#                             ylabel="length",
#                             curve_to_draw=[stats_opt.episode_lengths,
#                                            stats.episode_lengths],
#                             labels=["options", "q-learning"]
#                             )