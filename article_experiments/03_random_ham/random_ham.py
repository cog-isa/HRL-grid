from HAM.HAM_core import AbstractMachine, Start, Action, Stop, MachineRelation, MachineGraph, Choice, Call, RootMachine, LoopInvokerMachine
from HAM.HAM_experiments.HAM_utils import ham_runner, HAMParamsCommon
from article_experiments.global_envs import MazeEnvArticle, MazeEnvArticleSpecial, ArmEnvArticle, EnvironmentsArticle, get_cumulative_rewards
from environments.grid_maze_env.maze_world_env import MazeWorldEpisodeLength

name = "03_random_ham"




def run(global_env):
    if isinstance(global_env, ArmEnvArticle):
        pass
    elif isinstance(global_env, MazeEnvArticle):
        pass
    elif isinstance(global_env, MazeEnvArticleSpecial):
        pass
    else:
        raise KeyError
    full_name = name + "_" + global_env.__class__.__name__
    with open(full_name + " cumulative_reward.txt", "w") as w:
        for out in get_cumulative_rewards(rewards=rewards):
            w.write(str(out) + '\n', )

    with open(full_name + " reward.txt", "w") as w:
        for out in rewards:
            w.write(str(out) + '\n', )