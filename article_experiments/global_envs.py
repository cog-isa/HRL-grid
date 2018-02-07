from HAM.HAM_experiments.HAM_utils import maze_world_input_01
from environments.arm_env.arm_env import ArmEnv
from environments.grid_maze_env.grid_maze_generator import generate_maze_please, draw_maze
from environments.grid_maze_env.maze_world_env import MazeWorldEpisodeLength

action_minus_reward = -0.0001
finish_reward = 0.5


def get_cumulative_rewards(rewards):
    res = []
    current_cumulative_reward = 0
    for reward in rewards:
        current_cumulative_reward += reward
        res.append(current_cumulative_reward)
    return res

class EnvironmentsArticle:
    def __init__(self):
        self.environments = [
            MazeEnvArticle(),
            ArmEnvArticle(),
            MazeEnvArticleSpecial(),
        ]


class ArmEnvArticle:
    def __init__(self):
        self.env = ArmEnv(size_x=5, size_y=4, cubes_cnt=4, episode_max_length=500, finish_reward=finish_reward, action_minus_reward=action_minus_reward,
                          tower_target_size=4)
        self.episodes_count = 500


class MazeEnvArticle:
    def __init__(self):
        draw_maze(maze_world_input_01())
        self.env = MazeWorldEpisodeLength(maze=maze_world_input_01(), finish_reward=finish_reward, episode_max_length=500,
                                          wall_minus_reward=action_minus_reward * 5,
                                          action_minus_reward=action_minus_reward)
        self.episodes_count = 500


class MazeEnvArticleSpecial:
    def __init__(self):
        draw_maze(generate_maze_please(size_x=8, size_y=7))
        self.env = MazeWorldEpisodeLength(maze=generate_maze_please(size_x=8, size_y=7), finish_reward=finish_reward, episode_max_length=500,
                                          action_minus_reward=action_minus_reward, wall_minus_reward=action_minus_reward * 5)
        self.episodes_count = 500
