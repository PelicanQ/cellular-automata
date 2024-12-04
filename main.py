# Python code to implement Conway's Game Of Life
import matplotlib.axes
import matplotlib.lines
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import rules
from const import ON, OFF
import cluster
from scipy.ndimage import convolve
import matplotlib
import multiprocessing as mp
import time

# import numba

# setting up the values for the grid

vals = [ON, OFF]


def random_grid(N, p):
    """returns a grid of NxN random values

    Parameters:
        - N: grid size i.e. grid is NxN
        - p: probability for each cell to initially be ON
    """
    return np.random.choice(vals, N * N, p=[p, 1 - p]).reshape(N, N)


def start(N: int):
    """returns a grid of NxN values"""
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


chosen_rule = rules.fig6
rounds = 100
fractionsX = np.zeros((rounds, 1))
fractionsY = np.zeros((rounds, 1))
scatter: matplotlib.axes.Axes


def update(frame: int, grid, img=None):
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
                grid[i, j] = chosen_rule.birth_map[num_neighbors[i, j]]
            else:
                grid[i, j] = chosen_rule.death_map[num_neighbors[i, j]]

    if img:
        img.set_data(grid)

    # fractionsY[frame] = np.sum(grid) / (N * M * ON)  # set updated fraction
    # scatter.scatter(fractionsX[frame], fractionsY[frame])


def _update(_: int, grid, N: int, activity, img=None, act_img=None, fractions=None):
    """
    Returns: fraction of alive cells in the updated state
    """
    # copy grid since we require 8 neighbors
    # for calculation and we go line by line
    newGrid = grid.copy()
    for i in range(N):
        for j in range(N):

            # compute 8-neighbor sum
            # using toroidal boundary conditions - x and y wrap around
            # so that the simulation takes place on a toroidal surface.
            total = int(
                (
                    grid[i, (j - 1) % N]
                    + grid[i, (j + 1) % N]
                    + grid[(i - 1) % N, j]
                    + grid[(i + 1) % N, j]
                    + grid[(i - 1) % N, (j - 1) % N]
                    + grid[(i - 1) % N, (j + 1) % N]
                    + grid[(i + 1) % N, (j - 1) % N]
                    + grid[(i + 1) % N, (j + 1) % N]
                )
                / 255
            )

            # apply  rules

            # conway(i, j, total, grid, newGrid, activity)
            chosen_rule(i, j, total, grid, newGrid, activity)

    # update data
    grid[:] = newGrid[:]
    if img:
        img.set_data(newGrid)
    if act_img:
        act_img.set_data(activity)

    return np.sum(grid) / (N * N * ON)


def update_nowrap(frame, img, grid, N, activity, act_img):
    print(frame)
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


def animate(grid, *, update_interval=5, clusters=False):
    """Animate the cellular automata evolution.

    Parameters:
        - grid: initial grid
        - update_interval: animation update interval in milliseconds
        - clusters: bool to enable animated cluster statistics in a
        separate plot
    """

    # use the fact that False -> 0 and True -> 1 when adding
    num_plots = 1 + clusters

    # set up animation
    fig, axs = plt.subplots(ncols=num_plots)  # figure for CA grid

    # make the axs handle act the same for one and multiple axes
    if num_plots == 1:
        axs = (axs,)

    img = axs[0].imshow(grid, interpolation="nearest")
    # global scatter
    # scatter = axs[1]
    # t1 = time.perf_counter()
    # for i in range(100):
    #     update(i, grid, N, parallel=False)
    # t2 = time.perf_counter()
    # print(t2 - t1)

    # act_img = axs[2].imshow()
    # act_img.set_data(activity)
    img.set_data(grid)

    generation_text = axs[0].text(150, -40, '', fontsize=15, ha='center', va='center', color='black')

    if clusters:
        a = cluster.find_clusters(grid)
        u = np.unique(a, return_counts=True)
        axs[1].loglog(u[0], u[1])

    def anim_func(frame):
        update(frame, grid, img=img)

        generation_text.set_text(f"Generation: {frame}")

        if clusters:
            axs[1].clear()
            a = cluster.find_clusters(grid)
            u = np.unique(a, return_counts=True)
            axs[1].loglog(u[0], u[1])
            axs[1].set_xlabel("cluster size")
            axs[1].set_ylabel("frequency")

    anim = animation.FuncAnimation(
        fig,
        # lambda frame: update(frame, grid, parallel=False, img=img),
        lambda frame: anim_func(frame),
        interval=update_interval,
    )
    plt.show()


def main():
    N = 300
    grid = random_grid(N, 0.3)
    global chosen_rule
    chosen_rule = rules.conway
    animate(grid, clusters=True)


if __name__ == "__main__":
    main()
