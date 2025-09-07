import numpy as np
from matplotlib import pyplot as plt
import scienceplots

plt.style.use("science")


def plot():
    X = np.load("X.npy")
    Y = np.load("Y.npy")

    fig = plt.figure(figsize=(5.9, 3), tight_layout=True)
    ax1, ax2 = fig.subplots(1, 2)

    ax1.plot(X[: 2**12], Y[: 2**12], color="b")
    # ax2.plot([0, 2**16], [2**29, 2**29], color="gray", linestyle="--")

    ax2.plot(X, Y, color="b")
    ax2.plot([0, 2**16], [2**29, 2**29], color="gray", linestyle="--")

    ax1.set_xlabel("$y_n$")
    ax1.set_ylabel("$y_{n+1}$")
    ax2.set_xlabel("$y_n$")

    fig.savefig("figs/frac_conway.pdf", dpi=1000)

    plt.show()


if __name__ == "__main__":
    plot()
