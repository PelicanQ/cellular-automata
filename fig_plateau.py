import numpy as np
from matplotlib import pyplot as plt
from frac import num2grid


def show(start):

    fig = plt.figure(figsize=(16, 2))
    ax = fig.subplots()
    grid = num2grid(start).get()[-3:, :]
    img = ax.imshow(grid, cmap="gray_r")
    ax.grid(which="minor")
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    fig.savefig("figs/y0.pdf", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    # show(29000)
    # show(536879104)
    pass
