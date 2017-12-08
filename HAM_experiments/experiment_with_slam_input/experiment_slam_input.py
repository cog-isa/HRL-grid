from environments.maze_world_env import MazeWorldEpisodeLength
from lib import plotting
from HAM.utils import ham_learning
from HAM.machines import *
from environments.grid_maze_generator import *


def experiment_slam_input():
    from PIL import Image, ImageDraw
    im = Image.open('robots_map.jpg')
    img_drawer = ImageDraw.Draw(im)
    block_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 64, 80, 96, 120, 160, 192, 240, 320,
                   480, 960]
    block_size = block_sizes[6]
    n, m = im.height, im.width
    ss = set()
    for i in range(n):
        for j in range(m):
            q = sum(im.getpixel((i, j))) // 3
            offset = 253
            if q > offset:
                img_drawer.point((i, j), fill=(0, 0, 0))
            elif q > 50:
                img_drawer.point((i, j), fill=(255, 255, 255))
            else:
                img_drawer.point((i, j), fill=(0, 0, 0))

    N, M = n // block_size, m // block_size
    maze = np.zeros(shape=(N, M)).astype(int)

    for i in range(n // block_size):
        for j in range(m // block_size):
            ololo_sum = 0
            x, y = i, j
            for ii in range(x * block_size, x * block_size + block_size):
                for jj in range(y * block_size, y * block_size + block_size):
                    ololo_sum += sum(im.getpixel((ii, jj))) // 3

            ololo_sum /= block_size * block_size
            ss.add(ololo_sum)
            for ii in range(x * block_size, x * block_size + block_size):
                for jj in range(y * block_size, y * block_size + block_size):
                    if ololo_sum > 240:
                        maze[j][i] = 0
                    else:
                        maze[j][i] = 1
                    if ololo_sum > 240:
                        img_drawer.point((ii, jj), fill=(255, 255, 255))
                    else:
                        img_drawer.point((ii, jj), fill=(0, 0, 0))
    # im.save("new" + ".png")
    # exit(0)

    maze = place_start_finish(prepare_maze(maze))
    episode_max_length = 1000
    env = MazeWorldEpisodeLength(maze=maze, finish_reward=1000000, episode_max_length=episode_max_length)
    env.render()
    params = {
        "env": env,
        "num_episodes": 500,
        "machine": L2Interesting,
        "alpha": 0.1,
        "epsilon": 0.1,
        "discount_factor": 1,
        "path": []
    }
    Q1, stats1 = ham_learning(**params)
    plotting.plot_multi_test(curve_to_draw=[stats1.episode_rewards], smoothing_window=10)

    import imageio
    def PIL2array(img):
        return np.array(img.getdata(),
                        np.uint8).reshape(img.size[1], img.size[0], 3)

    im = Image.open('robots_map.jpg')

    d = params["path"][-episode_max_length:]
    images = []
    for index, item in enumerate(d):
        img_drawer = ImageDraw.Draw(im)
        y, x = item
        for ii in range(x * block_size, x * block_size + block_size):
            for jj in range(y * block_size, y * block_size + block_size):
                img_drawer.point((ii, jj), fill=(240, 13, 13))

        images.append(PIL2array(im))

        for ii in range(x * block_size, x * block_size + block_size):
            for jj in range(y * block_size, y * block_size + block_size):
                img_drawer.point((ii, jj), fill=(255, 255, 0))
        import time
        # if index > 100:
        #     break
    imageio.mimsave('movie.gif', images)


if __name__ == "__main__":
    experiment_slam_input()
