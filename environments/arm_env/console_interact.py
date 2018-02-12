from environments.arm_env.arm_env import ArmEnv

if __name__ == "__main__":

    print("\n" * 100)

    c_env = ArmEnv(size_x=3, size_y=3, cubes_cnt=2, episode_max_length=100, finish_reward=100, action_minus_reward=-1, tower_target_size=2)
    _, rew, is_done = (None, None, None)
    while not c_env.is_done():
        print('\n' * 100)
        c_env.render()

        print(rew, is_done)

        print("0 LEFT")
        print("1 UP")
        print("2 RIGHT")
        print("3 DOWN")
        print("4 ON")
        print("5 OFF")

        while True:
            try:
                act = int(input())
                break
            except ValueError:
                pass
        _, reward, done, _ = c_env.step(act)
