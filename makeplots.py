import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ca import CellularAutomata, Rule
from frac import frac
from rule90 import rule90


PLOT_DIR = Path("figs")


def make_plot_frac_conway():
    frac(ca=CellularAutomata(N=16, rule=Rule.CONWAY, wrap_boundary=False),
         ending=2**16,
         fig_save_path=PLOT_DIR/"frac_conway.pdf")


def make_plot_frac_Q():
    frac(ca=CellularAutomata(N=16, rule=Rule.Q, wrap_boundary=False),
         ending=2**16,
         fig_save_path=PLOT_DIR/"frac_Q.pdf")


def make_plot_rule90():
    x = np.arange(1000)
    y = np.array([rule90(num) for num in x])

    plt.plot(x/np.max(x), y/np.max(y))
    plt.xlabel("$y(j)$")
    plt.ylabel("$y(j+1)$")
    plt.savefig(PLOT_DIR/"rule90.pdf")


def main():
    try:
        import scienceplots
        plt.style.use('science')
    except ImportError:
        print("INFO: pip install scienceplots for nicer plots")

    if not PLOT_DIR.is_dir():
        print("making directory", PLOT_DIR.absolute())
        Path.mkdir(PLOT_DIR)

    make_plot_frac_conway()
    make_plot_frac_Q()
    make_plot_rule90()


if __name__ == "__main__":
    main()