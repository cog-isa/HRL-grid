import threading
from collections import defaultdict

import imageio
from pygame.time import delay

from HAM_old.machines_on_arm import BasicMachineArm, SmartMachine, SmartMachineTest
from environments.arm_env import ArmEnv
from utils import plotting
from HAM_old.utils import ham_learning
import numpy as np


def experiment_01():
    env = ArmEnv(size_x=4, size_y=3, cubes_cnt=3, episode_max_length=100, action_minus_reward=-1, finish_reward=100, tower_target_size=3)

    env.render()
    episodes_info = []
    params = {
        "env": env,
        "num_episodes": 1,
        "machine": BasicMachineArm,
        "alpha": 0.1,
        "epsilon": 0.1,
        "discount_factor": 0.9,
        "episodes_info": episodes_info
    }

    Q, stats = ham_learning(**params)
    to_draw_imgs = []
    for i in episodes_info[-1]:
        to_draw_imgs.append(env.render_to_image(i))
    imageio.mimsave('exp_01.gif', to_draw_imgs)
    plotting.plot_multi_test(curve_to_draw=[stats.episode_rewards])

    # img = env.render_to_image(env.get_evidence_for_image_render())
    # misc.imshow(img)
    # misc.imsave("image.png", img)


# noinspection PyTypeChecker
def experiment_02():
    def go_basic():
        env = ArmEnv(size_x=4, size_y=3, cubes_cnt=3, episode_max_length=100, action_minus_reward=-1, finish_reward=100, tower_target_size=3)

        env.render()
        episodes_info = []
        params = {
            "env": env,
            "num_episodes": 100,
            "machine": BasicMachineArm,
            "alpha": 0.1,
            "epsilon": 0.1,
            "discount_factor": 0.9,
            "episodes_info": episodes_info
        }

        Q, stats = ham_learning(**params)
        to_draw_imgs = []
        for i in episodes_info[-1]:
            to_draw_imgs.append(env.render_to_image(i))
        imageio.mimsave('exp_01.gif', to_draw_imgs)

    def go_smart():
        env = ArmEnv(size_x=4, size_y=3, cubes_cnt=3, episode_max_length=100, action_minus_reward=-1, finish_reward=100, tower_target_size=3)

        env.render()
        episodes_info = []
        params = {
            "env": env,
            "num_episodes": 100,
            "machine": SmartMachine,
            "alpha": 0.1,
            "epsilon": 0.1,
            "discount_factor": 0.9,
            "episodes_info": episodes_info
        }

        Q, stats = ham_learning(**params)
        to_draw_imgs = []
        for i in episodes_info[-1]:
            to_draw_imgs.append(env.render_to_image(i))
        imageio.mimsave('exp_02.gif', to_draw_imgs)

    t1 = threading.Thread(target=go_basic)
    t1.start()
    t2 = threading.Thread(target=go_smart)
    t2.start()


def experiment_03():
    episode_max_length = 200
    Q3 = defaultdict(lambda: defaultdict(lambda: 0))
    Q4 = defaultdict(lambda: defaultdict(lambda: 0))
    env = ArmEnv(episode_max_length=episode_max_length, size_x=4, size_y=3, cubes_cnt=3, action_minus_reward=-1, finish_reward=100, tower_target_size=3)
    episodes_count = 1200

    m3 = {"env": env, "num_episodes": episodes_count, "machine": SmartMachine, "q": Q3}
    m4 = {"env": env, "num_episodes": episodes_count, "machine": BasicMachineArm, "q": Q4}

    _, stats3 = ham_learning(**m3)
    _, stats4 = ham_learning(**m4)

    plotting.plot_multi_test(smoothing_window=30,
                             x_label="episode",
                             y_label="smoothed rewards",
                             curve_to_draw=[stats3.episode_rewards,
                                            stats4.episode_rewards,
                                            ],
                             labels=[
                                 m3["machine"],
                                 m4["machine"],
                             ])


def experiment_04():
    episode_max_length = 100
    Q3 = defaultdict(lambda: defaultdict(lambda: 0))
    env = ArmEnv(episode_max_length=episode_max_length, size_x=5, size_y=4, cubes_cnt=5, action_minus_reward=-1, finish_reward=100, tower_target_size=4)
    episodes_count = 1500

    episodes_info_3 = []
    # m3 = {"env": env, "num_episodes": episodes_count, "machine": BasicMachineArm, "q": Q3, "episodes_info": episodes_info_3}
    m3 = {"env": env, "num_episodes": episodes_count, "machine": SmartMachineTest, "q": Q3}

    _, stats3 = ham_learning(**m3)


    plotting.plot_multi_test(smoothing_window=30,
                             x_label="episode",
                             y_label="smoothed rewards",
                             curve_to_draw=[stats3.episode_rewards,
                                            ],
                             labels=[
                                 m3["machine"],
                             ])

    def save_image(name, data, fps):
        print(name)
        imageio.mimsave(name, data, fps=fps)

    threads = []
    for i in range(-3, 0):
        # data = list(map(env.render_to_image, episodes_info_3[i]))
        data = []
        for j in episodes_info_3[i]:
            data.append(env.render_to_image(j))
        t = threading.Thread(target=save_image, kwargs={"name": "exp_04_episode_0{i}.gif".format(**locals()), "data": data, "fps": 10})
        threads.append(t)
        t.start()
        # save_image('exp_04.gif', list(map(env.render_to_image, episodes_info_3[-1])), fps=10)
        # save_image('exp_05.gif', list(map(env.render_to_image, episodes_info_3[-2])), fps=10)
        # save_image('exp_06.gif', list(map(env.render_to_image, episodes_info_3[-3])), fps=10)
        # save_image('exp_07.gif', list(map(env.render_to_image, episodes_info_3[-4])), fps=10)
        # save_image('exp_08.gif', list(map(env.render_to_image, episodes_info_3[-5])), fps=10)


if __name__ == "__main__":
    experiment_04()
