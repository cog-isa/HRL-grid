from HAM.machines_on_arm import BasicMachineArm
from environments.arm_env import ArmEnv
from lib import plotting
from HAM.utils import ham_learning


def experiment_01():
    env = ArmEnv(size_x=4, size_y=3, cubes_cnt=3)

    env.render()
    params = {
        "env": env,
        "num_episodes": 550,
        "machine": BasicMachineArm,
        "alpha": 0.1,
        "epsilon": 0.1,
        "discount_factor": 0.9,
    }
    env.render_to_image()
    Q, stats = ham_learning(**params)
    plotting.plot_multi_test(curve_to_draw=[stats.episode_rewards])


if __name__ == "__main__":
    experiment_01()
