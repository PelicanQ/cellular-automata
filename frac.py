from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from ca import CellularAutomata, Rule, ON, OFF
#import rules
import csv


def num2grid(size: int, num: int):
    """
    takes an int, returns the corresponding grid
    num has to be 0 <= n < N*N

    """
    a = np.zeros((size * size,), dtype=np.int64)
    for i, bit in enumerate(bin(num)[2:].zfill(size * size)):
        a[i] = int(bit) * ON
    return np.reshape(a, shape=(size, size))


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


def frac(ca: CellularAutomata, ending: int, csv_save_path: str | Path | None = None, fig_save_path: str | Path | None = None) -> None:
    N = ca.N  # grid size
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

    for start_n in range(ending):
        # we choose one start and do one update

        print(f"Working... {(start_n / ending) * 100:.2f}% done", end="\r")
        
        ca.set_grid(num2grid(size=N, num=start_n))
        start_r, _ = grid2num(ca.grid, N)

        ca.update()
        # new_grid = update(grid=new_grid)

        new_r, new_n = grid2num(ca.grid, N)
        # if new_r < 1e-4:
        X.append(start_n)  # where we started
        Y.append(new_n)  # where update got us

    if csv_save_path:
        with open(csv_save_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows([X, Y])

    # fractal plot
    ax = plt.subplot(111)
    ax.plot(X, Y, color="b")
    # ax.plot([1, ending], [1, ending])
    ax.set_title("First order return map")
    ax.set_xlabel("$y_n$")
    ax.set_ylabel("$y_{n+1}$")
    # ax.set_ylim([-1, ending])
    ax.relim()
    ax.autoscale_view()

    if fig_save_path:
        plt.savefig(fig_save_path)

    plt.show()


if __name__ == "__main__":
    frac(ca=CellularAutomata(N=9, rule=Rule.CONWAY, wrap_boundary=False),
         ending=2**16,
         csv_save_path="map.csv")