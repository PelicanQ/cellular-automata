import numpy as np
from matplotlib import pyplot as plt
from calc_dim import fit_sticks

# The weierstrass function graph is a fractal. Find its fractal dim
a = 0.5
b = 7


def weier(x, N):
    tot = np.zeros((len(x),))
    for n in range(N):
        tot += a**n * np.cos(b**n * x)
    return tot


if __name__ == "__main__":
    xx = np.linspace(0, 10, 10000000)
    yy = weier(xx, 300)
    stick_lengths = np.linspace(1, 0.1, 10)
    stick_nums, fits_x, fitsy = fit_sticks(xx, yy, stick_lengths, 1e-100, 1e-100)
    plt.plot(xx, yy)

    y = np.log10(stick_nums)
    x = -np.log10(stick_lengths)
    p = np.polyfit(x, y, 1)
    print(p)
    plt.plot(x, y)
    plt.show()
