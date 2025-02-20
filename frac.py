from matplotlib import pyplot as plt
import numpy as np
from const import OFF, ON
from main import update
import csv


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


def frac():
    N = 9  # grid size
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
    # g = num2grid(N, n=7)
    # ng = update(g, 0)
    # ax.imshow(ng)
    # plt.show()
    ending = 2**9
    for start_n in range(ending):
        # we choose one start and do one update
        print(f"{(start_n / ending):.3f}")
        grid = num2grid(size=N, num=start_n)
        start_r, _ = grid2num(grid, N)

        update(grid=grid)
        # new_grid = update(grid=new_grid)

        new_r, new_n = grid2num(grid, N)
        # if new_r < 1e-4:
        X.append(start_n)  # where we started
        Y.append(new_n)  # where update got us
        # return

    with open("map.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([X, Y])

    # fractal plot
    ax = plt.subplot(111)
    ax.plot(X, Y, color="b")
    # ax.plot([1, ending], [1, ending])
    ax.set_title("First order return map")
    ax.set_xlabel("y_n")
    ax.set_ylabel("y_n+1")
    # ax.set_ylim([-1, ending])
    ax.relim()
    ax.autoscale_view()
    plt.show()


if __name__ == "__main__":
    frac()
