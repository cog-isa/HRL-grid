import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from copy import deepcopy
from queue import Queue


def generate_pattern(pattern_id):
    assert (0 <= pattern_id < 2 ** 13)
    pattern = np.zeros(shape=(5, 5)).astype(int)
    pattern[0, 0] = pattern[0, 1] = pattern[1, 0] = 1
    pattern[4, 0] = pattern[3, 0] = pattern[4, 1] = 1
    pattern[0, 4] = pattern[0, 3] = pattern[1, 4] = 1
    pattern[4, 4] = pattern[3, 4] = pattern[4, 3] = 1

    bit_cnt = 1
    for i in range(5):
        for j in range(5):
            if pattern[i][j] == 1:
                continue
            if (pattern_id & bit_cnt) > 0:
                pattern[i][j] = 1
            bit_cnt *= 2
    return pattern


def draw_maze(maze):
    cmap = colors.ListedColormap(['white', 'darkgreen'])
    bounds = [0, 1, 40]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    ax.imshow(maze, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.show()


def generate_maze(blocks, size_x, size_y, seed=314159265):
    np.random.seed(seed)
    maze = None
    for i in range(size_y):
        row = blocks[np.random.choice(len(blocks), size=1, replace=False)[0]]
        for j in range(1, size_x):
            row = np.concatenate((row, blocks[np.random.choice(len(blocks), 1, replace=False)[0]]), axis=1)
        maze = np.concatenate((maze, row), axis=0) if maze is not None else row

    return maze


def prepare_maze(maze):
    maze_size_x, maze_size_y = len(maze), len(maze[0])
    for i in range(maze_size_x):
        maze[i][0] = maze[i][maze_size_y - 1] = 1
    for i in range(maze_size_y):
        maze[0][i] = maze[maze_size_x - 1][i] = 1

    return maze


def bfs(start_x, start_y, maze):
    maze_size_x, maze_size_y = len(maze), len(maze[0])

    # max value
    MX = (maze_size_x + 10) * (maze_size_y + 10)
    maze[maze == 0] = MX
    q = Queue()
    q.put((start_x, start_y))
    maze[start_x, start_y] = 0
    while not q.empty():
        x, y = q.get()
        assert (maze[x][y] != -1)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx * dy != 0 or (dx == 0 and dy == 0):
                    continue
                nx, ny = x + dx, y + dy
                if maze[nx][ny] == -1:
                    continue
                if maze[nx][ny] > maze[x][y]:
                    q.put((nx, ny))
                    maze[nx][ny] = maze[x][y] + 1
    maze[maze == MX] = 0


def place_start_finish(maze):
    mz = deepcopy(maze)
    maze_size_x, maze_size_y = len(mz), len(mz[0])
    mz[mz > 0] = -1
    for i in range(maze_size_x):
        for j in range(maze_size_y):
            if mz[i][j] == 0:
                bfs(start_x=i, start_y=j, maze=mz)
    max_dist = int(np.max(mz))

    for i in range(maze_size_x):
        for j in range(maze_size_y):
            if mz[i][j] == max_dist:
                start_x, start_y = i, j
                mz[mz > 0] = 0
                bfs(start_x=start_x, start_y=start_y, maze=mz)
                max_dist = int(np.max(mz))
                for ii in range(maze_size_x):
                    for jj in range(maze_size_y):
                        if mz[ii][jj] == max_dist:
                            finish_x, finish_y = ii, jj
                            maze[start_x, start_y] = 2
                            maze[finish_x, finish_y] = 3
                            return maze
    raise ValueError


def generate_maze_please(size_x=3, size_y=4):
    t = place_start_finish(prepare_maze(generate_maze(blocks=[generate_pattern(64)], size_x=size_x, size_y=size_y)))
    return t


if __name__ == "__main__":
    # a = generate_maze(blocks=[generate_pattern(i) for i in [7]], size_x=2, size_y=2)
    # b = generate_maze(blocks=[generate_pattern(i) for i in [64, 64]], size_x=2, size_y=2)
    # t = prepare_maze(generate_maze(blocks=[generate_pattern(64)], size_x=3, size_y=4))
    # t = place_start_finish(t)
    # print(t)
    t = generate_maze_please()
    # print("******" * 10)
    print(t)
