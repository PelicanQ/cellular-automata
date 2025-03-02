import pandas as pd
import matplotlib.pyplot as plt


def make_plot_frac_conway_zoom():
    df = pd.read_csv("data" / "conway16nowrap.csv", index_col=None, header=None)
    X = df.iloc[0]
    Y = df.iloc[1]
    cut = 4100
    ratio = 2
    wid = 6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(wid, wid / ratio))
    ax2.plot(X, Y, color="b")
    ax2.set_xlabel("$y_n$")
    ax1.set_xlabel("$y_n$")
    ax1.set_ylabel("$y_{n+1}$")
    ax1.plot(X[:cut], Y[:cut], color="b")
    fig.tight_layout()
    plt.savefig("fig" / "frac_conway_zoom.svg")
    plt.show()
