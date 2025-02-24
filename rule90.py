import numpy as np
import matplotlib.pyplot as plt


def rule90(num):
    """Maps an int to an int by applying Rule 90 to its binary representation.

    For info on Rule 90, see: https://en.wikipedia.org/wiki/Rule_90
    """
    s = np.binary_repr(num)
    n = len(s)

    current_state = np.zeros((n,), dtype=int)

    for i, bit in enumerate(s):
        current_state[i] = int(bit)


    next_state = np.zeros((n,), dtype=int)

    for i, _ in enumerate(current_state):
        if i > 0 and i < n-1:
            next_state[i] = int(np.logical_xor(current_state[i-1], current_state[i+1]))
        else:
            next_state[i] = current_state[i]

    next_num = sum(1<<i for i, b in enumerate(next_state[::-1]) if b)

    return next_num


def main():
    x = np.arange(1000)
    y = np.array([rule90(num) for num in x])
    
    plt.plot(x,y)
    plt.xlabel("$y(j)$")
    plt.ylabel("$y(j+1)$")
    plt.show()


if __name__ == "__main__":
    main()