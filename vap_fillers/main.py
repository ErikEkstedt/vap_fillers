import matplotlib.pyplot as plt
import numpy as np

from vap.utils import read_txt
from vap_fillers.dialog_act_filler_matching import plot_filler_da_loc_distribution


def calc_cross_diff(fillers, col_name="NOW_CROSS"):
    columns = [
        "SESSION",
        "FILLE-ID",
        "S",
        "E",
        "SPEAKER",
        "UH_OR_UM",
        "WITH_OR_OMIT_FILLER",
        "NOW_CROSS",
        "FUT_CROSS",
        "FILLER_F0",
        "FILLER_INT",
    ]
    col2idx = {k: v for v, k in enumerate(columns)}
    cross = fillers[:, col2idx[col_name]].astype(float)
    idx = fillers[:, col2idx["FILLE-ID"]].astype(int)
    wfill = cross[::2]
    idx = idx[::2]
    ofill = cross[1::2]
    cross = cross[1::2]  # when does ofill cross?
    diff = wfill - ofill
    return diff, idx, cross


def plot_diff_hist(diff):
    scaled_diff = diff / 50
    cutoff = 0.2
    fig, ax = plt.subplots(1, 1)
    plt.hist(
        scaled_diff[scaled_diff < -cutoff],
        bins=100,
        range=(-6, 6),
        color="r",
        label="Filler worse?",
    )
    plt.hist(
        scaled_diff[scaled_diff > cutoff],
        bins=100,
        range=(-6, 6),
        color="g",
        label="Filler better?",
    )
    plt.hist(
        scaled_diff[np.abs(scaled_diff) <= cutoff],
        bins=100,
        range=(-6, 6),
        color="b",
        label="SAME",
    )
    ax.legend()
    plt.tight_layout()
    return fig


if __name__ == "__main__":

    fillers = np.array([f.split(",") for f in read_txt("data/test_fillers_dal.txt")])

    fig, _ = plot_filler_da_loc_distribution(fillers)
    plt.show()

    d = read_txt("results/fillers/filler_output.txt")
    fillers = np.array([f.split() for f in d])
    header = list(fillers[0])
    fillers = fillers[1:]

    diff_now, idx_now, cross_now = calc_cross_diff(fillers, col_name="NOW_CROSS")
    diff_fut, idx_fut, cross_fut = calc_cross_diff(fillers, col_name="FUT_CROSS")
    diff, idx, cross = diff_now, idx_now, cross_now

    fig = plot_diff_hist(diff)
    plt.show()

    # run
    # streamlit run vap_fillers/visualize_filler.py
    # look at interesting idx in browser
    # Where P_NOW_WITHOUT crosses before a 'short time' 1, 2 s ? and the difference is largs -> strongest fillers

    # to seconds
    # focus_cond = diff <= -100
    focus_cond = np.logical_and(diff >= 100, diff != 500)
    focus_cond = np.logical_and(focus_cond, cross <= 50)
    focus_cond = np.logical_and(focus_cond, cross > 0)

    ii = np.where(focus_cond)[0]
    focus_ids = idx[ii]
    focus_diffs = diff[ii]
    focus_cross = cross[ii]
    n = len(focus_ids)
    for i, d, c in zip(focus_ids[:n], focus_diffs[:n], focus_cross[:n]):
        print(i, f"{d} ({c})")
        input()
