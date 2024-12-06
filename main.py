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


birth_map = [OFF, OFF, OFF, ON, OFF, OFF, OFF, OFF, OFF]  # if dead
death_map = [OFF, OFF, ON, ON, OFF, OFF, OFF, OFF, OFF]  # if alive


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


# def hamming(grid: np.ndarray, grid2: np.ndarray | None = None):
#     # calculate hamming distance to zero for now
#     a = grid.flatten()
#     d = scipy.spatial.distance.hamming(a, b)
#     return d
def num2grid(N: int, n: int):
    """n has to be 0 <= n < N*N"""
    a = np.zeros((N * N,), dtype=np.int16)
    for i, bit in enumerate(bin(n)[2:].zfill(N * N)):
        a[i] = int(bit) * ON
    return np.reshape(a, newshape=(N, N))


def main():
    N = 16  # grid size
    FRAMES = 20  # how many updates for one run
    RUNS = 10  # how many inital conditions
    updateInterval = 300  # milliseconds animation update
    rs = np.zeros((RUNS, FRAMES))  # real number encoding state

    # grid = start3(N)
    # grid2 = start4(N)
    # for run in range(RUNS):
    # for frame in range(FRAMES):
    X = []
    Y = []
    ending = 2**17
    for start_n in range(ending):
        print(f"{(start_n / ending):.3f}")
        grid = num2grid(N=N, n=start_n)
        new1 = update(grid=grid)
        flat1 = new1.flatten()
        r = 0
        for i, digit in enumerate(flat1):
            n = (digit // 255) * pow(2, -i)
            r += n
        X.append(start_n)  # where we started
        Y.append(r)  # where update got us

    print("DONE")
    # Y = rs[1:]
    # X = rs[:-1]
    ax = plt.subplot(111)
    ax.plot(X, Y, marker=".")
    ax.set_title("First order return map")
    ax.set_xlabel("y_n")
    ax.set_ylabel("y_n+1")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    plt.show()
    # inits plots
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)  # figure for CA grid

    # img: matplotlib.image.AxesImage = ax1.imshow(grid, interpolation="nearest")
    # img2: matplotlib.image.AxesImage = ax3.imshow(grid2, interpolation="nearest")

    # dist: matplotlib.axes.Axes = ax2
    # dist2: matplotlib.axes.Axes = ax4
    # [hamming_line] = dist.plot(0, 0, label="Hamming", marker=".", color="b")
    # [jac_line] = dist.plot(0, 0, marker=".", color="r", label="Jaccard")
    # [dice_line] = dist.plot(0, 0, marker=".", color="g", label="Dice")
    # [kul_line] = dist.plot(0, 0, marker=".", color="b", label="Kul")
    # [line1] = dist.plot(0, 0, marker=".", color="y", label="Line1")
    # [line2] = dist.plot(0, 0, marker=".", label="Line2")
    # [line3] = dist.plot(0, 0, marker=".", label="Line3")
    # # [line4] = dist2.plot(0, 1, marker=".", label="Line4")
    # # [map_line] = dist2.plot()
    # dist.set_xlim([0, FRAMES])
    # dist.legend()

    # anim = animation.FuncAnimation(
    #         fig,
    #         update_all,
    #         interval=updateInterval,
    #         frames=FRAMES,
    #         repeat=False,
    #     )
    # plt.show()
    # def one_run():

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

    # flat1 = new1.flatten()
    # flat2 = new2.flatten()
    # r = 0
    # for i, digit in enumerate(flat1):
    #     n = (digit // 255) * pow(2, -i)
    #     r += n

    # rs[frame] = r
    # # print(rs)
    # add_linedata(
    #     hamming_line, flat1, flat2, scipy.spatial.distance.hamming, frame
    # )
    # add_linedata(jac_line, flat1, flat2, scipy.spatial.distance.jaccard, frame)
    # add_linedata(
    #     dice_line,
    #     flat1,
    #     flat2,
    #     lambda a, b: np.abs(scipy.spatial.distance.dice(a, b)),
    #     frame,
    # )
    # add_linedata(
    #     kul_line, flat1, flat2, scipy.spatial.distance.kulczynski1, frame
    # )
    # add_linedata(
    #     line1, flat1, flat2, scipy.spatial.distance.rogerstanimoto, frame
    # )
    # add_linedata(line2, flat1, flat2, scipy.spatial.distance.russellrao, frame)
    # add_linedata(
    #     line3, flat1, flat2, scipy.spatial.distance.sokalmichener, frame
    # )
    # add_linedata(line4, flat1, flat2, scipy.spatial.distance.sokalsneath, frame)

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
