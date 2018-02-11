from HAM.HAM_core import RootMachine, LoopInvokerMachine
from HAM.HAM_experiments.HAM_utils import HAMParamsCommon, ham_runner
from HAM.HAM_experiments.experiment_04_auto_random_HAM_on_maze_env.experiment_04 import create_random_machine
from article_experiments.global_envs import MazeEnvArticleSpecial, ArmEnvArticle, MazeEnvArticle, \
    get_cumulative_rewards, EnvironmentsArticle
from utils.graph_drawer import draw_graph

name = "05_random"


def run(global_env):
    rewards = None
    if isinstance(global_env, ArmEnvArticle):
        pass
    elif isinstance(global_env, MazeEnvArticle):
        pass
    elif isinstance(global_env, MazeEnvArticleSpecial):
        env = global_env.env
        seed = 573846788
        internal_machine = create_random_machine(maximal_number_of_vertex=6, maximal_number_of_edges=6,
                                                 random_seed=seed,
                                                 env=env)
        machine = RootMachine(machine_to_invoke=LoopInvokerMachine(machine_to_invoke=internal_machine))
        draw_graph(file_name="maze_env_special",
                   graph=internal_machine.get_graph_to_draw(action_to_name_mapping=env.get_actions_as_dict()))
        params = HAMParamsCommon(env)
        ham_runner(ham=machine, num_episodes=global_env.episodes_count, env=env, params=params)
        rewards = params.logs["ep_rewards"]
    else:
        raise KeyError

    if rewards is not None:
        full_name = name + "_" + global_env.__class__.__name__
        # with open(full_name + " cumulative_reward.txt", "w") as w:
        #     for out in get_cumulative_rewards(rewards=rewards):
        #         w.write(str(out) + '\n', )

        with open(full_name + " reward.txt", "w") as w:
            for out in rewards:
                w.write(str(out) + '\n', )


def main():
    for global_env in EnvironmentsArticle().environments:
        run(global_env)


if __name__ == '__main__':
    main()
