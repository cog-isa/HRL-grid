from DQN.arm_env_dqn import ArmEnvDQN

if __name__ == "__main__":

    print("\n" * 100)

    c_env = ArmEnvDQN(episode_max_length=200,
                 size_x=5,
                 size_y=4,
                 cubes_cnt=3,
                 scaling_coeff=3,
                 action_minus_reward=-1,
                 finish_reward=200,
                 tower_target_size=3)
    _, reward, done = (None, None, None)
    done = False
    while not done:
        print('\n' * 100)
        c_env.render()

        print(reward, done)

        print("0 LEFT")
        print("1 UP")
        print("2 RIGHT")
        print("3 DOWN")
        print("4 TOGGLE")
        # print("5 OFF")

        while True:
            try:
                act = int(input())
                break
            except ValueError:
                pass
        _, reward, done, _ = c_env.step(act)