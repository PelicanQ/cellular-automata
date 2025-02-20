# Python code to implement Conway's Game Of Life
import matplotlib.axes
import matplotlib.image
import matplotlib.lines
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.spatial
import rules
from const import ON, OFF
import cluster
from scipy.ndimage import convolve
import scipy
import matplotlib
import csv

# setting up the values for the grid

vals = [ON, OFF]


def randomGrid(N, p):
    """returns a grid of NxN random values"""
    return np.random.choice(vals, N * N, p=[p, 1 - p]).reshape(N, N)


def start(N: int):
    if N % 2 == 1:
        raise Exception("N must be even")
    a = np.zeros((N, N), dtype=np.int16)
    m = int(N / 2)
    a[m, m] = ON
    a[m + 1, m] = ON
    a[m - 1, m] = ON
    a[m, m + 1] = ON
    a[m + 1, m + 1] = ON
    a[m - 1, m + 1] = ON
    a[m, m - 1] = ON
    a[m + 1, m - 1] = ON
    a[m - 1, m - 1] = ON

    return a


def start2(N: int):
    if N % 2 == 1:
        raise Exception("N must be even")
    a = np.zeros((N, N), dtype=np.int16)
    m = int(N / 2) + 2
    a[m, m] = ON
    a[m + 1, m] = ON
    a[m - 1, m] = ON
    a[m, m + 1] = ON
    a[m + 1, m + 1] = ON
    a[m - 1, m + 1] = ON
    a[m, m - 1] = ON
    a[m + 1, m - 1] = ON
    a[m - 1, m - 1] = ON

    # pertubation
    # a[m - 2, m - 1] = ON
    # a[m - 2, m] = ON
    return a


def start3(N: int):
    if N % 2 == 1:
        raise Exception("N must be even")
    a = np.zeros((N, N), dtype=np.int16)
    m = int(N / 2) + 2
    a[m - 0, m + 1] = ON
    a[m - 1, m] = ON
    a[m - 2, m] = ON
    a[m - 3, m + 1] = ON
    a[m - 2, m + 2] = ON
    a[m - 1, m + 2] = ON

    # pertub
    a[m - 0, m] = ON
    a[m, m - 1] = ON
    return a


def start4(N: int):
    if N % 2 == 1:
        raise Exception("N must be even")
    a = np.zeros((N, N), dtype=np.int16)
    m = int(N / 2) + 2
    a[m - 0, m + 1] = ON
    a[m - 1, m] = ON
    a[m - 2, m] = ON
    a[m - 3, m + 1] = ON
    a[m - 2, m + 2] = ON
    a[m - 1, m + 2] = ON

    # pertub
    a[m - 0, m] = ON
    a[m + 1, m - 1] = ON
    return a


chosen_rule = rules.conway
rounds = 100
fractionsX = np.zeros((rounds, 1))
fractionsY = np.zeros((rounds, 1))
scatter: matplotlib.axes.Axes


birth_map = [OFF, OFF, OFF, ON, OFF, OFF, OFF, OFF, OFF]  # if dead OG
death_map = [OFF, OFF, ON, ON, OFF, OFF, OFF, OFF, OFF]  # if alive OG
# birth_map = [OFF, OFF, OFF, ON, ON, OFF, OFF, OFF, OFF]  # if dead
# death_map = [OFF, OFF, ON, ON, OFF, OFF, OFF, OFF, OFF]  # if alive


def update(
    grid,
    frame: int | None = None,
    img=None,
    parallel=False,
    line: matplotlib.lines.Line2D | None = None,
):
    """Returns grid"""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int16)
    conv = convolve(grid, kernel, mode="wrap")
    num_neighbors = np.floor_divide(conv, 255)

    # num rows
    N = np.shape(grid)[0]

    # num cols
    M = np.shape(grid)[1]

    # fractionsX[frame] = np.sum(grid) / (N * M * ON)  # set prev fraction

    # update grid according to global chosen_rule

    for i in range(N):
        for j in range(M):
            if grid[i, j] == OFF:
                grid[i, j] = birth_map[num_neighbors[i, j]]
            else:
                grid[i, j] = death_map[num_neighbors[i, j]]
            # grid[i, j] = chosen_rule(grid, i, j, int(num_neighbors[i, j]))
    return grid
    if img:
        img.set_data(grid)
    if line:
        b = line.get_data()
        n0 = np.append(b[0], frame)
        n1 = np.append(b[1], hamming(grid))
        line.set_data(n0, n1)
        print(line.get_data())

    # fractionsY[frame] = np.sum(grid) / (N * M * ON)  # set updated fraction
    # scatter.scatter(fractionsX[frame], fractionsY[frame])


def update_nowrap(frame, img, grid, N, activity, act_img):
    # copy grid since we require 8 neighbors
    # for calculation and we go line by line
    newGrid = grid.copy()
    # corners
    total = grid[1, 0] + grid[1, 1] + grid[0, 1]
    chosen_rule(0, 0, total, grid, newGrid, activity)

    total = grid[N - 1, 1] + grid[N - 2, 1] + grid[N - 2, 0]
    chosen_rule(N - 1, 0, total, grid, newGrid, activity)

    total = grid[0, N - 2] + grid[1, N - 1] + grid[1, N - 2]
    chosen_rule(0, N - 1, total, grid, newGrid, activity)

    total = grid[N - 2, N - 1] + grid[N - 1, N - 2] + grid[N - 2, N - 2]
    chosen_rule(N - 1, N - 1, total, grid, newGrid, activity)

    # side up
    i = 0
    for j in range(1, N - 1):
        total = int(
            (
                grid[0, (j - 1)]
                + grid[0, (j + 1)]
                + grid[1, (j - 1)]
                + grid[1, (j)]
                + grid[1, (j + 1)]
            )
            / 255
        )
        chosen_rule(i, j, total, grid, newGrid, activity)
    # side down
    i = N - 1
    for j in range(1, N - 1):
        total = int(
            (
                grid[N - 1, (j - 1)]
                + grid[N - 1, (j + 1)]
                + grid[N - 2, (j - 1)]
                + grid[N - 2, (j)]
                + grid[N - 2, (j + 1)]
            )
            / 255
        )
        chosen_rule(i, j, total, grid, newGrid, activity)

    # side left
    j = 0
    for i in range(1, N - 1):
        total = int(
            (
                grid[i + 1, 0]
                + grid[i - 1, 0]
                + grid[i - 1, 1]
                + grid[i, 1]
                + grid[i + 1, 1]
            )
            / 255
        )
        chosen_rule(i, j, total, grid, newGrid, activity)

    # side right
    j = N - 1
    for i in range(1, N - 1):
        total = int(
            (
                grid[i + 1, N - 1]
                + grid[i - 1, N - 1]
                + grid[i - 1, N - 2]
                + grid[i, N - 2]
                + grid[i + 1, N - 2]
            )
            / 255
        )
        chosen_rule(i, j, total, grid, newGrid, activity)

    # inner points
    for i in range(1, N - 1):
        for j in range(1, N - 1):

            # compute 8-neighbor sum
            # using toroidal boundary conditions - x and y wrap around
            # so that the simulation takes place on a toroidal surface.
            total = int(
                (
                    grid[i, (j - 1)]
                    + grid[i, (j + 1)]
                    + grid[(i - 1), j]
                    + grid[(i + 1), j]
                    + grid[(i - 1), (j - 1)]
                    + grid[(i - 1), (j + 1)]
                    + grid[(i + 1), (j - 1)]
                    + grid[(i + 1), (j + 1)]
                )
                / 255
            )

            # apply  rules

            # conway(i, j, total, grid, newGrid, activity)
            chosen_rule(i, j, total, grid, newGrid, activity)

    # update data
    img.set_data(newGrid)
    act_img.set_data(activity)
    grid[:] = newGrid[:]


def num2grid(size: int, num: int):
    """
    takes an int, returns the corresponding grid
    num has to be 0 <= n < N*N

    """
    a = np.zeros((size * size,), dtype=np.int64)
    for i, bit in enumerate(bin(num)[2:].zfill(size * size)):
        a[i] = int(bit) * ON
    return np.reshape(a, newshape=(size, size))


def grid2num(grid: np.ndarray, N):
    flat1 = grid.flatten()
    r = 0
    b = ""
    for i, digit in enumerate(flat1):
        b += f"{digit//255}"
        # n = (digit // 255) * pow(2, -i)
        # r += n
    # print(b)
    # print(" ")
    n = int(b, 2)
    r = n / (2 ** (N * N))
    return r, n


def main():
    N = 4  # grid size
    # Grid size | Fractal depth, Times to zoom
    # 8 5
    # 9 6
    # 11 8
    # 15 12
    # yep its minus 3
    # inits plots
    # fig, ax = plt.subplots(ncols=1, nrows=1)  # figure for CA grid
    X = []
    Y = []
    QX = []
    QY = []
    # g = num2grid(N, n=7)
    # ng = update(g, 0)
    # ax.imshow(ng)
    # plt.show()
    ending = 2**16
    for start_n in range(ending):
        # we choose one start and do one update
        print(f"{(start_n / ending):.3f}")
        grid = num2grid(size=N, num=start_n)
        start_r, _ = grid2num(grid, N)

        new_grid = update(grid=grid)
        new_grid = update(grid=new_grid)

        Q, Q_n = grid2num(new_grid, N)
        QX.append(start_n)  # where we started
        QY.append(Q_n)  # where update got us

        new_grid = update(grid=new_grid)
        new_grid = update(grid=new_grid)
        # new_grid = update(grid=new_grid)
        # new_grid = update(grid=new_grid)
        # new_grid = update(grid=new_grid)
        # new_grid = update(grid=new_grid)
        # new_grid = update(grid=new_grid)
        # new_grid = update(grid=new_grid)

        new_r, new_n = grid2num(new_grid, N)
        # if new_r < 1e-4:
        X.append(start_n)  # where we started
        Y.append(new_n)  # where update got us
        # return

    with open("map.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([X, Y])

    # fractal plot
    ax = plt.subplot(111)
    ax.plot(X, Y, linewidth=0, marker=".", color="b")
    ax.plot(QX, QY, linewidth=0, marker="x", color="g")
    ax.plot([1, ending], [1, ending])
    ax.set_title("First order return map")
    ax.set_xlabel("y_n")
    ax.set_ylabel("y_n+1")
    ax.set_ylim([-1, ending])
    ax.relim()
    ax.autoscale_view()
    plt.show()

    # dist.set_xlim([0, FRAMES])
    #     img.set_data(grid)
    #     img2.set_data(grid2)
    #     def add_linedata(line, flat1, flat2, dist_func, frame):
    #         data = line.get_data()
    #         dist = dist_func(flat1, flat2)
    #         log = np.append(data[1], dist)
    #         line.set_data(np.append(data[0], frame), log)

    #     def update_all(frame: int):
    # new1 = update(frame, grid, parallel=False, img=img)
    # new2 = update(frame, grid2, parallel=False, img=img)
    # img.set_data(new1)
    # img2.set_data(new2)

    # print(np.max(X), np.min(X))
    # dist.relim()
    # dist.autoscale_view()
    # dist2.relim()
    # dist2.autoscale_view()
    # if frame == FRAMES - 1:
    # done()


# call main
if __name__ == "__main__":
    main()

# if hopefully_start_n != start_n:
#     # something went wrong. This is a check that back and forth conversion works
#     exit("AAAAAGHHHH")
# Gostick J, Khan ZA, Tranter TG, Kok MDR, Agnaou M, Sadeghi MA, Jervis R. PoreSpy: A Python Toolkit for Quantitative Analysis of Porous Media Images. Journal of Open Source Software, 2019. doi:10.21105/joss.01296
