from environments.arm_env.arm_env import ArmEnvToggleTopOnly


def main():
    def get_on_model(self):
        return self.get_arm_x(), self.is_cube_graped()

    def get_arm_x(self):
        return self._size_x - self._arm_x

    def is_cube_graped(self):
        cube_dx, cube_dy = self.MOVE_ACTIONS[self.ACTIONS.DOWN]
        cube_x, cube_y = self._arm_x + cube_dx, self._arm_y + cube_dy
        return self._magnet_toggle and self.ok(cube_x, cube_y) and self._grid[cube_x][cube_y] == 1

    ArmEnvToggleTopOnly.get_arm_x = get_arm_x
    ArmEnvToggleTopOnly.is_cube_graped = is_cube_graped
    ArmEnvToggleTopOnly.get_on_model = get_on_model

    env = ArmEnvToggleTopOnly(size_x=3, size_y=3, cubes_cnt=2, episode_max_length=300, finish_reward=100, action_minus_reward=0.001, tower_target_size=2)
    print("height:", env.get_arm_x())
    print("cube_graped:"
          , env.is_cube_graped())
    print(env.get_on_model())


if __name__ == '__main__':
    main()
