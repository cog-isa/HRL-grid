from SearchHie.main import environments
from environments.arm_env.arm_env import ArmEnv, ArmEnvToggle, ArmEnvToggleTopOnly

if __name__ == "__main__":

    print("\n" * 100)

    c_env = environments[3]
    _, rew, is_done = (None, None, None)
    done = False
    while not done:
        print('\n' * 100)
        c_env.render()

        print(rew, is_done)

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