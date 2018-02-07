from article_experiments.global_envs import MazeEnvArticle, MazeEnvArticleSpecial, ArmEnvArticle, EnvironmentsArticle, get_cumulative_rewards
from environments.weak_methods import q_learning

name = "01_table_q-learning"





def run(global_env):
    full_name = name + "_" + global_env.__class__.__name__
    rewards, _ = q_learning(env=global_env.env, num_episodes=global_env.episodes_count)

    with open(full_name + " cumulative_reward.txt", "w") as w:
        for out in get_cumulative_rewards(rewards=rewards):
            w.write(str(out) + '\n', )

    with open(full_name + " reward.txt", "w") as w:
        for out in rewards:
            w.write(str(out) + '\n', )


def main():
    for global_env in EnvironmentsArticle().environments:
        run(global_env)


if __name__ == '__main__':
    main()
