# Python code to implement Conway's Game Of Life
import numpy as np
# import matplotlib.lines
# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cluster
from scipy.ndimage import convolve
import time
# import numba
from enum import IntEnum


# Cell states
ON = 255
OFF = 0


class CellularAutomataRule:
    def __init__(self, birth_map, death_map) -> None:
        # lists where the index is a cells number of neighbors and
        # the value is the state of the cell in the next generation
        self._birth_map = birth_map  # if dead
        self._death_map = death_map  # if alive

    @property
    def birth_map(self):
        return self._birth_map

    @property
    def death_map(self):
        return self._death_map


class Rule:
    CONWAY = CellularAutomataRule(
        birth_map=[OFF, OFF, OFF, ON, OFF, OFF, OFF, OFF, OFF],
        death_map=[OFF, OFF, ON, ON, OFF, OFF, OFF, OFF, OFF],
    )

    FIG6 = CellularAutomataRule(
        birth_map=[OFF, OFF, OFF, ON, OFF, OFF, ON, OFF, OFF],
        death_map=[OFF, OFF, ON, ON, OFF, ON, ON, OFF, OFF],
    )

    FIG8 = CellularAutomataRule(
        birth_map=[OFF, OFF, OFF, OFF, ON, OFF, OFF, OFF, OFF],
        death_map=[OFF, OFF, ON, ON, ON, ON, ON, OFF, OFF],
    )

    Q = CellularAutomataRule(
        birth_map=[OFF, OFF, ON, OFF, ON, OFF, ON, OFF, OFF],
        death_map=[OFF, OFF, OFF, ON, ON, ON, OFF, OFF, OFF],
    )


class CellularAutomata:
    def __init__(self, N, rule, wrap_boundary=True):
        self.rule = rule
        self.N = N
        self.wrap = wrap_boundary
        self.grid = np.zeros((N, N), dtype=np.int16)

    def set_grid(self, grid):
        self.grid = grid

    def init_random(self, p):
        """Initialize a grid of random values.

        Parameters:
            - p: probability for each cell to initially be ON
        """
        self.grid = np.random.choice([ON, OFF], self.N * self.N, p=[p, 1 - p]).reshape(self.N, self.N)

    def update(self, img=None):
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int16)
        conv = convolve(self.grid, kernel, mode="wrap" if self.wrap else "constant", cval=OFF)
        num_neighbors = np.floor_divide(conv, 255)

        # num rows
        N = np.shape(self.grid)[0]

        # num cols
        M = np.shape(self.grid)[1]

        # update grid according to global self.rule
        for i in range(N):
            for j in range(M):
                if self.grid[i, j] == OFF:
                    self.grid[i, j] = self.rule.birth_map[num_neighbors[i, j]]
                else:
                    self.grid[i, j] = self.rule.death_map[num_neighbors[i, j]]

        if img:
            img.set_data(self.grid)


# def start(N: int):
#     """returns a grid of NxN values (initial values for fig 6)"""
#     if N % 2 == 1:
#         raise Exception("N must be even")
#     a = np.zeros((N, N), dtype=np.int16)
#     m = int(N / 2)
#     a[m, m] = ON
#     a[m + 1, m] = ON
#     a[m - 1, m] = ON
#     a[m, m + 1] = ON
#     a[m + 1, m + 1] = ON
#     a[m - 1, m + 1] = ON
#     a[m, m - 1] = ON
#     a[m + 1, m - 1] = ON
#     a[m - 1, m - 1] = ON

#     return a


def animate(ca: CellularAutomata, *, update_interval: int = 5, show_cluster_stats: bool = False):
    """Animate the cellular automata evolution.

    Parameters:
        - ca: cellular automata with an initial grid
        - update_interval: animation update interval in milliseconds
        - show_cluster_stats: bool to enable animated cluster statistics in a
        separate plot
    """

    # use the fact that False -> 0 and True -> 1 when adding
    num_plots = 1 + show_cluster_stats

    # set up animation
    fig, axs = plt.subplots(ncols=num_plots)  # figure for CA grid

    # make the axs handle act the same for one and multiple axes
    if num_plots == 1:
        axs = (axs,)

    img = axs[0].imshow(ca.grid, interpolation="nearest")
    img.set_data(ca.grid)

    # TODO: coords should be function of grid size and window size
    generation_text = axs[0].text(
        150, -40, "", fontsize=15, ha="center", va="center", color="black"
    )

    def plot_clusters():
        size, freq = cluster.get_cluster_dist(ca.grid)
        axs[1].loglog(size, freq)
        axs[1].set_xlabel("cluster size")
        axs[1].set_ylabel("frequency")

    if show_cluster_stats:
        plot_clusters()

    def anim_func(frame):
        ca.update(img=img)

        generation_text.set_text(f"Generation: {frame}")

        if show_cluster_stats:
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

    ca = CellularAutomata(N, rule, wrap_boundary=True)
    cluster_list = []

    for i in range(num_seeds):
        print(f"Working... {i/num_seeds * 100:.1f}% done", end="\r")
        ca.init_random(p)

        for _ in range(num_generations):
            ca.update()

        a = cluster.find_clusters(ca.grid)
        cluster_list.append(a)

    clusters = np.concatenate(cluster_list)

    size, freq = np.unique(clusters, return_counts=True)
    freq_normalized = freq / np.sum(freq)

    _, ax = plt.subplots()
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
        rule=Rule.FIG8,
        N=100,
        p=0.2,
        num_seeds=1000,
        num_generations=200,
        savepath="plots/fig8.pdf",
    )


def make_plot_cluster_dist_conway():
    plot_cluster_dist(
        rule=Rule.CONWAY,
        N=100,
        p=0.2,
        num_seeds=1000,
        num_generations=2000,
        savepath="plots/conway-100-02-1000-2000.pdf",
    )


def make_plot_cluster_dist_fig6():
    plot_cluster_dist(
        rule=Rule.FIG6,
        N=100,
        p=0.2,
        num_seeds=1000,
        num_generations=100,
        savepath="plots/fig6-100-02-1000-100.pdf",
    )


def demo_animate():
    ca = CellularAutomata(N=100, rule=Rule.FIG6, wrap_boundary=True)
    ca.init_random(p=0.2)
    animate(ca, show_cluster_stats=True)


def demo_plot_cluster_dist():
    plot_cluster_dist(Rule.CONWAY, 100, 0.2, 10, 200)


def main():
    demo_animate()


if __name__ == "__main__":
    main()
