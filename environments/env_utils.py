from HAM.HAM_experiments.HAM_utils import maze_world_input_01
from environments.arm_env.arm_env import ArmEnv
from environments.grid_maze_env.grid_maze_generator import draw_maze, prepare_maze, generate_pattern
from environments.grid_maze_env.maze_world_env import MazeWorldEpisodeLength


class EnvForTesting:
    def __init__(self):
        self.env = MazeWorldEpisodeLength(maze=maze_world_input_01(), finish_reward=2000, episode_max_length=400)
        # self.env = ArmEnv(size_x=5, size_y=4, cubes_cnt=4, episode_max_length=300, finish_reward=20000, action_minus_reward=-1, tower_target_size=4)
        self.episodes = 1600


class EnvForTesting2:
    def __init__(self):
        self.env = ArmEnv(size_x=4, size_y=3, cubes_cnt=3, episode_max_length=300, finish_reward=20000, action_minus_reward=-1, tower_target_size=3)
        self.episodes = 1800


def main():
    draw_maze(prepare_maze(maze_world_input_01()))


if __name__ == '__main__':
    main()
