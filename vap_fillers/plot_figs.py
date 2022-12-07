import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from os.path import join
from vap_fillers.main import load_fillers


def plot_df(df):
    fig, ax = plt.subplots(2, 1)
    pdf["a_f0_mean"].hist(bins=100, ax=ax[0])
    pdf["b_f0_mean"].hist(bins=100, ax=ax[0])
    pdf["a_intensity_mean"].hist(bins=100, ax=ax[1])
    pdf["b_intensity_mean"].hist(bins=100, ax=ax[1])
    plt.show()

    # Raw filler
    fig, ax = plt.subplots(2, 1)
    df["filler_f0_m"].hist(bins=100, ax=ax[0], label="F0", color="b")
    df["filler_intensity_m"].hist(bins=100, ax=ax[1], label="Intensity", color="g")
    ax[0].legend()
    ax[1].legend()
    plt.show()

    # Standardize
    f0 = (df["filler_f0_m"] - df["speaker_f0_m"]) / df["speaker_f0_s"]
    ii = (df["filler_intensity_m"] - df["speaker_intensity_m"]) / df[
        "speaker_intensity_s"
    ]
    fig, ax = plt.subplots(2, 1)
    f0.hist(bins=100, range=(-5, 5), ax=ax[0], color="b", label="F0")
    ii.hist(bins=100, range=(-5, 5), ax=ax[1], color="g", label="Intensity")
    ax[0].legend()
    ax[1].legend()
    plt.show()


def plot_prosody(df, fig_root=None, plot=True):
    fig, ax = plt.subplots(2, 1)
    df["filler_f0_m"].hist(bins=100, ax=ax[0], alpha=0.5, color="g", label="F0")
    df["filler_intensity_m"].hist(
        bins=100, ax=ax[1], alpha=0.5, color="b", label="Intensity"
    )
    ax[0].legend()
    ax[1].legend()
    fig.suptitle("Prosody raw")
    plt.tight_layout()
    if fig_root is not None:
        fig.savefig(join(fig_root, "prosody.png"))
    if plot:
        plt.pause(0.1)
    return fig


def plot_prosody_z(df, fig_root=None, plot=True):
    fig, ax = plt.subplots(2, 1)
    df["filler_f0_z"].hist(
        bins=100, range=(-5, 5), ax=ax[0], alpha=0.5, color="g", label="F0 Z"
    )
    df["filler_intensity_z"].hist(
        bins=100, range=(-3, 3), ax=ax[1], alpha=0.5, color="b", label="Intensity Z"
    )
    ax[0].legend()
    ax[1].legend()
    fig.suptitle("Prosody z-normalized")
    plt.tight_layout()
    if fig_root is not None:
        fig.savefig(join(fig_root, "prosody_z.png"))
    if plot:
        plt.pause(0.1)
    return fig


def plot_diff(df, min_cross=0, max_cross=500, fig_root=None, plot=True):
    fc = df[df["filler_now_cross"] > min_cross]
    fc = fc[fc["filler_now_cross"] < max_cross]
    fc = fc[fc["omit_now_cross"] < max_cross]
    fc = fc[fc["omit_now_cross"] > min_cross]

    tot = len(fc)

    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    fc["filler_now_cross"].hist(
        bins=100, ax=ax[0], color="b", alpha=0.5, label="Filler cross"
    )
    fc["omit_now_cross"].hist(
        bins=100, ax=ax[1], color="g", alpha=0.5, label="Omit cross"
    )
    fc["diff"].hist(bins=100, ax=ax[2], color="r", alpha=0.5, label="Diff cross")
    fm = fc["filler_now_cross"].mean()
    om = fc["omit_now_cross"].mean()
    bm = fc["diff"].mean()
    ax[0].axvline(fm, color="k", linewidth=2, label=f"Mean: {round(fm)}")
    ax[1].axvline(om, color="k", linewidth=2, label=f"Mean: {round(om)}")
    ax[2].axvline(bm, color="k", linewidth=2, label=f"Mean: {round(bm)}")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    fig.suptitle(f"Cross N={tot} ({min_cross} < cross < {max_cross})")
    plt.tight_layout()

    if fig_root is not None:
        fig.savefig(join(fig_root, "diff.png"))

    if plot:
        plt.pause(0.1)
    return fig


def plot_omitted_cross_filler_no_cross(df, fig_root=None, plot=True):
    d = df[df["filler_now_cross"] == -1]
    d = d[d["omit_now_cross"] > -1]
    n = len(d)
    fig, ax = plt.subplots(1, 1)
    d["omit_now_cross"].hist(
        bins=100, ax=ax, color="orange", label=f"omitted cross ({n})"
    )
    fig.suptitle("Omitted cross when filler did NOT cross")
    plt.tight_layout()

    if fig_root is not None:
        fig.savefig(join(fig_root, "omitted_when_filler_nocross.png"))

    if plot:
        plt.pause(0.1)
    return fig


def get_bar_stats(df, filler_type="filler", n_frames_instant=5):
    tot = len(df)
    # remove those which do not terminate
    d = df[df[f"{filler_type}_now_cross"] != -1]
    n_nc = tot - len(d)
    tot = len(d)
    # remove those that terminate quickly
    d = d[d[f"{filler_type}_now_cross"] > n_frames_instant]
    n_instant = tot - len(d)
    # get mean/std of remaining points
    cross_m = d[f"{filler_type}_now_cross"].mean()
    cross_s = d[f"{filler_type}_now_cross"].std()
    return {
        "no_cross": n_nc,
        "instant": n_instant,
        "other_cross_m": cross_m,
        "other_cross_s": cross_s,
        "n_frames_instant": n_frames_instant,
    }


def get_filler_no_cross_but_omit_do(df):
    l = np.logical_and(df["filler_now_cross"] == -1, df["omit_now_cross"] != -1)
    return len(df[l])


def plot_event_bars(df, n_instant=0, fig_root=None, plot=True):
    fill = get_bar_stats(df, n_frames_instant=n_instant)
    omit = get_bar_stats(df, filler_type="omit", n_frames_instant=n_instant)
    n_oc_fn = get_filler_no_cross_but_omit_do(df)

    y = (
        100
        * np.array(
            [
                fill["no_cross"],
                omit["no_cross"],
                fill["instant"],
                omit["instant"],
                n_oc_fn,
            ]
        )
        / len(df)
    )
    xlabels = ["F-no-cross", "O-no-cross", "F-instant", "O-instant", "O-cross F-not"]

    fig, ax = plt.subplots(1, 1)
    a = ax.bar(xlabels, y, color=["b", "g", "b", "g", "orange"], alpha=0.5)
    ax.legend(a[-3:], ["Filler", "Omit", "both"])
    ax.set_ylabel("Percentage %")
    fig.suptitle(f"Events N={len(df)} (instant <= {n_instant})")
    plt.tight_layout()
    if fig_root is not None:
        fig.savefig(join(fig_root, "event_bars.png"))
    if plot:
        plt.pause(0.1)
    return fig


if __name__ == "__main__":

    df = load_fillers("results/all_fillers_test_prosody_model.csv")

    fig_root = "results/images"
    Path(fig_root).mkdir(parents=True, exist_ok=True)

    _ = plot_prosody(df, fig_root=fig_root, plot=False)
    _ = plot_prosody_z(df, fig_root=fig_root, plot=False)
    _ = plot_diff(df, min_cross=10, fig_root=fig_root, plot=False)
    _ = plot_omitted_cross_filler_no_cross(df, fig_root=fig_root, plot=False)
    _ = plot_event_bars(df, n_instant=10, fig_root=fig_root)
