# from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import cupy as cp
import cupyx.scipy.ndimage as cpndimage

from line_profiler import profile

SIZE = 21
powers_two_core = (1 << cp.arange(SIZE * SIZE - 1, -1, -1)).reshape(
    (SIZE, SIZE)
)  # [2^(N*N)-1 2^(N*N-1)-1 2^(N*N-2)-1 ... 4 2 1]
powers_two = cp.zeros((SIZE + 2, SIZE + 2), dtype=cp.float64)
powers_two[1:-1, 1:-1] = powers_two_core

shifts = cp.arange(SIZE * SIZE - 1, -1, -1, dtype=int).reshape(
    (SIZE, SIZE)
)  # [N*N-1 N*N-2 ... 3 2 1 ]
zero_padded = cp.zeros((SIZE + 2, SIZE + 2), dtype=cp.float64)  # pad with zeros
num_neighbors = cp.copy(zero_padded)
kernel = cp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=cp.float64)


@cp.fuse()
def totalPoly(n, x):
    """n: number neighbors. x: cell value"""
    dead = (
        -n * (n - 1) * (n - 2) * (n - 4) * (n - 5) * (n - 6) * (n - 7) * (n - 8) // 720
    )
    live = (
        n * (n - 1) * (n - 3) * (n - 4) * (n - 5) * (n - 6) * (n - 7) * (n - 8) // 1440
    ) + (
        n
        * (n - 1)
        * (n - 2)
        * (n - 4)
        * (n - 5)
        * (n - 6)
        * (n - 7)
        * (n - 8)
        // (-720)
    )
    return x * (live - dead) + dead


@cp.fuse()
def num2grid(num: int):
    """
    takes an int, returns the corresponding grid
    num has to be 0 <= n < N*N
    """
    return cp.bitwise_and(num >> shifts, 1)


def grid2num(grid: cp.ndarray):
    """Input grid padded with zeros"""
    n = cp.sum(grid * powers_two)
    # r = n / (2 ** (N * N))
    return None, n


arange = np.arange(SIZE - 1, -1, -1)


def add_one(grid):
    for i in arange:
        row = grid[i, :].get()
        for j in arange:
            row[j] = 1 - row[j]
            if row[j] == 1:
                grid[i, :] = cp.asarray(row)
                return
        grid[i, :] = cp.asarray(row)

    raise Exception("How did I get here?")


grid_core = cp.zeros((SIZE, SIZE), dtype=cp.float64)


# @profile
def frac(
    ending: int,
    # csv_save_path: str | Path | None = None,
    # fig_save_path: str | Path | None = None,
) -> None:
    global grid_core
    X = []
    Y = []
    for start_n in range(ending):
        # we choose one start and do one update
        if start_n % 1000 == 0:
            print(f"Working... {(start_n / ending) * 100:.2f}% done", end="\r")
        zero_padded[1:-1, 1:-1] = grid_core

        def cov():
            return cpndimage.convolve(zero_padded, kernel)

        num_neighbors = cov()
        grid = totalPoly(num_neighbors, zero_padded)
        _, new_n = grid2num(grid)

        # increase grid
        add_one(grid_core)

        X.append(start_n)  # where we started
        Y.append(int(new_n))  # where update got us

    np.save(f"X_{SIZE}", np.array(X))
    np.save(f"Y_{SIZE}", np.array(Y))


def plot():
    X = np.load(f"X_{SIZE}.npy")
    Y = np.load(f"Y_{SIZE}.npy")
    ax = plt.subplot(111)
    ax.plot(X, Y, color="b")
    ax.set_title("First order return map")
    ax.set_xlabel("$y_n$")
    ax.set_ylabel("$y_{n+1}$")
    # ax.plot([0, 2**SIZE], [2**29, 2**29], color="gray", linestyle="--")
    # ax.plot([0, 2**SIZE], [2**30, 2**30])
    ax.relim()
    ax.autoscale_view()
    # ax.plot([0, 2**SIZE], [2**16, 2**16])
    # if fig_save_path:
    #     plt.savefig(fig_save_path)

    plt.show()


if __name__ == "__main__":
    # frac(2**SIZE)
    # show(29000, 30429)
    # 28669 - 30429
    plot()
    # frac(
    #     ca=CellularAutomata(N=SIZE, rule=Rule.CONWAY, wrap_boundary=False),
    #     ending=2**SIZE,
    #     # csv_save_path="map.csv",
    # )
