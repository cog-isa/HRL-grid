import numpy as np
from Options_new.maze_options import MazeWorldTrain, place_finish, prepare_train_maze, Option
import sys
from environments.maze_world_env import MazeWorld, MazeWorldEpisodeLength
from lib import plotting
import itertools

from environments.grid_maze_generator import place_start_finish, generate_maze, generate_pattern, prepare_maze

#create environment
ids = [2, 32, 64, 1024]
blocks = [generate_pattern(i) for i in ids]
maze, pattern_no_to_id = generate_maze(blocks=blocks, size_x=4, size_y=5, options=True)
m = place_start_finish(prepare_maze(maze))
env = MazeWorldEpisodeLength(maze=m, episode_max_length=700)

env.render()

# for each type of pattern create 4 train environments
train_envs = np.array([[MazeWorldTrain(place_finish(prepare_train_maze(generate_pattern(id)), direction=a))
               for a in range(4)] for id in ids])

# for each type of pattern we have 4 options (even if some ways are blocked)
maze_options = np.array([[Option(train_env_).learning_option() for train_env_ in train_env] for train_env in train_envs])


def q_learning_on_options(env, pattern_no_to_id, options, num_episodes, eps=0.6, alpha=0.1, gamma=0.99):
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
            if i_episode == num_episodes - 1:
                print("\n")
                print(state)
                env.render()

            # Take an action
            if np.random.rand(1) < eps:  # choose random option
                action0 = np.random.choice(np.arange(n_actions + n_options), size=1)[0]
            else:
                action0 = np.argmax(q_table[state, :])

            if i_episode == num_episodes - 1: print(action0)

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


stats_opt, q_table_opt = q_learning_on_options(env, pattern_no_to_id, maze_options, 3000)
plotting.plot_episode_stats(stats_opt)
