import numpy as np
from matplotlib import pyplot as plt
import numba

UPDATE = False

if UPDATE:
    xx = [0]
    yy = [0]
    plt.ion()
    fig, ax = plt.subplots(1, 1)
    [line] = ax.plot(xx, yy, marker=".")
    plt.show()


def update(x, y):
    if not UPDATE:
        return
    print(x, y)
    plt.pause(0.1)
    xx.append(x)
    yy.append(y)
    line.set_data(xx, yy)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()


def find_intersection(x, y, xs, ys, i, r, tol_y):
    """When a stick reaches beyond a point, this finds the last point it reaches to"""
    p_prev = np.array([x, y])
    p_iminus1 = np.array([xs[i - 1], ys[i - 1]])
    p_i = np.array([xs[i], ys[i]])

    if (
        np.abs(p_prev[1] - p_iminus1[1]) < tol_y
        and np.abs(p_iminus1[1] - p_i[1]) < tol_y
    ):
        # we have three points with same y value. Treat separately to avoid numeric problems
        return x + r, y

    # find intersection between circle traced by stick and line from point i-1 to point i
    v1 = p_prev - p_iminus1
    v2 = p_i - p_iminus1
    c = np.linalg.norm(v1)
    q = np.linalg.norm(v2)
    cos_alpha = np.dot(v1, v2) / (c * q)
    t = c * cos_alpha + np.sqrt(c**2 * cos_alpha**2 - c**2 + r**2)
    p_intersection = p_iminus1 + t * v2 / q
    return p_intersection[0], p_intersection[1]


def stick(x1, y1, x2, y2, r, tol_r):
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    if np.abs(r - dist) < tol_r:
        # snap to point
        return "snap"
    elif r < dist:
        # stick is smaller
        return "less"
    else:
        # reaches beyond
        return "beyond"


# @numba.jit
def calc_num_sticks(
    xs: np.array, ys: np.array, length: float, tol_r: float, tol_y: float
):
    r = length
    prev_x = float(xs[0])
    prev_y = float(ys[0])
    update(prev_x, prev_y)

    fit_x: list[float] = [prev_x]
    fit_y: list[float] = [prev_y]
    num_sticks = 0
    i = 1
    while i < len(xs):
        # we try to reach point i with our stick
        # print(i, num_sticks)
        x = xs[i]
        y = ys[i]

        s = stick(prev_x, prev_y, x, y, r, tol_r)

        if s == "snap":
            # snap to point
            prev_x = x
            prev_y = y
            update(prev_x, prev_y)

            fit_x.append(prev_x)
            fit_y.append(prev_y)
            num_sticks += 1
            if i == len(xs) - 1:
                # we snapped to final point
                return num_sticks, fit_x, fit_y
            i += 1
            continue
        elif s == "less":
            # entire stick fits along first line.
            dx = x - prev_x
            dy = y - prev_y
            dist = np.sqrt(dx**2 + dy**2)

            m = 1  # num sticks to lay on this line
            n = int(np.floor(dist / r))  # number of sticks which fit
            if n > 10:
                # we fit many sticks in one go.
                m = n - 2  # For safety, add a little fewer than all that may fit

            prev_x += m * dx * r / dist
            prev_y += m * dy * r / dist
            update(prev_x, prev_y)

            fit_x.append(prev_x)
            fit_y.append(prev_y)
            num_sticks += m
            continue
        else:
            # stick reaches further than point i
            while s == "beyond":
                if i == len(xs) - 1:
                    # stick reaches beyond the last point
                    return num_sticks, fit_x, fit_y
                i += 1
                s = stick(prev_x, prev_y, xs[i], ys[i], r, tol_r)

            # i is now the point we reach for
            if s == "snap":
                prev_x = xs[i]
                prev_y = ys[i]
                if i == len(xs) - 1:
                    # snap to last point
                    fit_x.append(prev_x)
                    fit_y.append(prev_y)
                    num_sticks += 1
                    return num_sticks, fit_x, fit_y
                i += 1
            else:
                # Our stick cannot reach point i but it can reach a point on the line from i-1 to i
                intersect_x, intersect_y = find_intersection(
                    prev_x, prev_y, xs, ys, i, r, tol_y
                )

                prev_x = intersect_x
                prev_y = intersect_y

            num_sticks += 1
            fit_x.append(prev_x)
            fit_y.append(prev_y)
            update(prev_x, prev_y)
            continue  # move onto next
    raise Exception("This is unexpected")


def fit_sticks(xs, ys, stick_lengths, tol_r, tol_y):
    """Arrays have to have dtype float!"""
    fits_x: list[list[float]] = []
    fits_y: list[list[float]] = []
    nums = np.zeros((len(stick_lengths),), dtype=int)
    for i, leng in enumerate(stick_lengths):
        nums[i], fit_x, fit_y = calc_num_sticks(xs, ys, leng, tol_r, tol_y)
        fits_x.append(fit_x)
        fits_y.append(fit_y)
    return nums, fits_x, fits_y


def see():
    """see the lines it draws"""
    X = np.load("X_20.npy").astype(float)
    Y = np.load("Y_20.npy").astype(float)
    ax.plot(X, Y)

    n, fx, fy = calc_num_sticks(X, Y, 1e11, 1e-6, 1e-2)
    # n, fx, fy = calc_num_sticks(X, Y, 1e10, 1e-6, 1e-2)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # see()
    # exit()
    X = np.load("X_21.npy").astype(float)
    Y = np.load("Y_21.npy").astype(float)
    stick_lengths = np.linspace(5e11, 1e10, 10)

    stick_nums, fits_x, fits_y = fit_sticks(X, Y, stick_lengths, 1e-2, 1e-2)

    y = np.log10(stick_nums)
    x = -np.log10(stick_lengths)
    p = np.polyfit(x, y, 1)
    print(p)
    plt.plot(x, y)
    plt.show()
