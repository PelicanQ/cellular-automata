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
        - N: grid size, i.e. grid is NxN
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


def update(frame: int, grid, img=None, wrap: bool = True):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int16)
    conv = convolve(grid, kernel, mode="wrap" if wrap else "constant", cval=OFF)
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

    # TODO: coords should be function of grid size and window size
    generation_text = axs[0].text(
        150, -40, "", fontsize=15, ha="center", va="center", color="black"
    )

    def plot_clusters():
        size, freq = cluster.get_cluster_dist(grid)
        axs[1].loglog(size, freq)
        axs[1].set_xlabel("cluster size")
        axs[1].set_ylabel("frequency")

    if clusters:
        plot_clusters()

    def anim_func(frame):
        update(frame, grid, img=img)

        generation_text.set_text(f"Generation: {frame}")

        if clusters:
            axs[1].clear()
            plot_clusters()

    anim = animation.FuncAnimation(
        fig,
        # lambda frame: update(frame, grid, parallel=False, img=img),
        lambda frame: anim_func(frame),
        interval=update_interval,
    )
    plt.show()


def plot_cluster_dist(
    rule, N=100, p=0.2, num_seeds=10, num_generations=100, savepath=None
):
    """Plot the distribution of cluster sizes for an evolution rule.

    Parameters:
        - rule: the rule to update the grid with
        - N: grid size, i.e. grid is NxN
        - p: probability for each cell to initially be ON
        - num_seeds: the number of random initial seeds/grids to use
        - num_generations: the number of generations to run each initial grid through
        - savepath: the plot is saved to this path if specified, else it isn't saved
    """

    t1 = time.perf_counter()

    global chosen_rule
    chosen_rule = rule

    cluster_list = []

    for i in range(num_seeds):
        print(f"Working... {i/num_seeds * 100:.1f}% done", end="\r")
        grid = random_grid(N, p)

        for j in range(num_generations):
            update(j, grid, None)

        a = cluster.find_clusters(grid)
        cluster_list.append(a)

    clusters = np.concatenate(cluster_list)

    size, freq = np.unique(clusters, return_counts=True)
    freq_normalized = freq / np.sum(freq)

    fig, ax = plt.subplots()
    ax.loglog(size, freq_normalized)
    ax.set_xlabel("cluster size")
    ax.set_ylabel("frequency")

    t2 = time.perf_counter()
    print(f"Completed after {t2 - t1} ns")

    if savepath:
        plt.savefig(savepath)

    plt.show()


def recreate_fig8():
    plot_cluster_dist(
        rule=rules.fig8,
        N=100,
        p=0.2,
        num_seeds=1000,
        num_generations=200,
        savepath="plots/fig8.pdf",
    )


def make_plot_cluster_dist_conway():
    plot_cluster_dist(
        rule=rules.conway,
        N=100,
        p=0.2,
        num_seeds=1000,
        num_generations=2000,
        savepath="plots/conway-100-02-1000-2000.pdf",
    )


def make_plot_cluster_dist_fig6():
    plot_cluster_dist(
        rule=rules.fig6,
        N=100,
        p=0.2,
        num_seeds=1000,
        num_generations=100,
        savepath="plots/fig6-100-02-1000-100.pdf",
    )


def demo_plot_cluster_dist():
    plot_cluster_dist(rules.conway, 100, 0.2, 10, 2000)


def demo_animate():
    N = 100
    grid = random_grid(N, 0.2)
    global chosen_rule
    chosen_rule = rules.fig6
    animate(grid, clusters=True)


def main():
    demo_animate()


if __name__ == "__main__":
    main()
