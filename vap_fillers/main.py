import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


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


def combine_da_and_prosody_info():

    df = pd.read_csv(
        "data/test_fillers_dal.txt",
        names=["SESSION", "S", "E", "SPEAKER", "UH_OR_UM", "LOC", "WORDS_IN_DA", "DA"],
    )
    df = df.sort_values("SESSION")
    df2 = pd.read_csv("results/fillers/filler_output.txt", sep=" ")

    columns = list(df2.columns)
    new_cols = columns + ["LOC", "WORDS_IN_DA", "DA"]

    skipped = 0
    sk = []
    df_new = pd.DataFrame(columns=new_cols)
    for ii in tqdm(range(len(df)), desc="Combine prosody + DA info"):
        da_info = df.loc[ii, :]  # get the DA entry
        # get the subset for that session
        session_subset = df2[df2["SESSION"] == da_info["SESSION"]]
        # Get
        curr = session_subset.loc[lambda x: np.abs(x["S"] - da_info["S"]) < 0.02]
        # curr = session_subset.loc[lambda x: x["S"]== da_info["S"]]
        if len(curr) > 0:
            row = curr[curr["WITH_OR_OMIT_FILLER"] == 1].copy()
            nofill = curr[curr["WITH_OR_OMIT_FILLER"] == 0].copy()
            # c = row["NOW_CROSS"].values
            # nfc = nofill["NOW_CROSS"].values
            row["DIFF_NOW_CROSS"] = row["NOW_CROSS"].values - nofill["NOW_CROSS"].values
            row["DIFF_FUT_CROSS"] = row["FUT_CROSS"].values - nofill["FUT_CROSS"].values
            row["LOC"] = da_info["LOC"]
            row["WORDS_IN_DA"] = da_info["WORDS_IN_DA"]
            row["DA"] = da_info["DA"]
            df_new = pd.concat([df_new, row])
        else:
            # print("Skipped")
            # print(da_info)
            # print(curr)
            sk.append(da_info)
            skipped += 1
            # input()
    print("Skipped: ", skipped)

    df_new.to_csv("data/all_filler_info.csv", index=False)


def plot_prosody(df, xmin=-6, xmax=3, n_bins=50, min_duration=0.11, plot=True):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    um_f0 = (
        df.loc[:, "f0"].loc[df["uh_or_um"] == "um"].loc[df["duration"] >= min_duration]
    )
    uh_f0 = (
        df.loc[:, "f0"].loc[df["uh_or_um"] == "uh"].loc[df["duration"] >= min_duration]
    )
    um_in = (
        df.loc[:, "intensity"]
        .loc[df["uh_or_um"] == "um"]
        .loc[df["duration"] >= min_duration]
    )
    uh_in = (
        df.loc[:, "intensity"]
        .loc[df["uh_or_um"] == "uh"]
        .loc[df["duration"] >= min_duration]
    )
    um_f0.hist(bins=n_bins, color="b", range=(xmin, xmax), ax=ax[0, 0], label="UM F0")
    uh_f0.hist(bins=n_bins, color="r", range=(xmin, xmax), ax=ax[1, 0], label="UH F0")
    ax[0, 0].axvline(0, color="k", linewidth=2, linestyle="dashed")
    ax[1, 0].axvline(0, color="k", linewidth=2, linestyle="dashed")
    ax[0, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_xlim([xmin, xmax])
    um_in.hist(bins=n_bins, color="b", range=(xmin, xmax), ax=ax[0, 1], label="UM I")
    uh_in.hist(bins=n_bins, color="r", range=(xmin, xmax), ax=ax[1, 1], label="UH I")
    ax[0, 1].axvline(0, color="k", linewidth=2, linestyle="dashed")
    ax[1, 1].axvline(0, color="k", linewidth=2, linestyle="dashed")
    ax[0, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_xlim([xmin, xmax])
    for aa in ax:
        for a in aa:
            a.legend()
    if plot:
        plt.pause(0.1)
    return fig, ax


def plot_diff(df, min_frame_cross=0, max_frame_cross=498, n_bins=50, plot=True):
    """"""
    # condition to only use entries
    # min_frame_cross <= entry-cross <= max_frame_cross
    condition = min_frame_cross <= df["filler_now_cross"].values
    condition = np.logical_and(condition, df["filler_now_cross"] <= max_frame_cross)

    fnc = df[condition]["filler_now_cross"]
    nnc = df[condition]["omit_now_cross"]
    dd = df[condition]["diff"]

    # fnc = df[df["filler_now_cross"] >= min_frame_cross]["filler_now_cross"]
    # nnc = df[df["omit_now_cross"] >= min_frame_cross]["omit_now_cross"]
    # dd = df[df["omit_now_cross"] >= min_frame_cross]["diff"]
    fig, ax = plt.subplots(3, 1, figsize=(8, 8))
    fig.suptitle(f"Diff (min_frame_cross: {min_frame_cross})")
    fnc.hist(bins=n_bins, ax=ax[0], color="b", label="Filler")
    ax[0].axvline(fnc.mean(), color="k", label=f"Filler mean {round(fnc.mean(), 1)}")
    nnc.hist(bins=n_bins, ax=ax[1], color="g", label="Omitted")
    ax[1].axvline(nnc.mean(), color="k", label=f"Omitted mean {round(nnc.mean(), 1)}")
    dd.hist(bins=n_bins, ax=ax[2], color="r", label="Diff")
    ax[2].axvline(dd.mean(), color="k", label=f"diff mean {round(dd.mean(), 1)}")
    for a in ax:
        a.legend()
    plt.tight_layout()

    if plot:
        plt.pause(0.1)
    return fig, ax


if __name__ == "__main__":
    df = pd.read_csv("results/filler_info/filler_info.csv")
    df["diff"] = df.loc[:, "filler_now_cross"] - df.loc[:, "omit_now_cross"]
    df["duration"] = df.loc[:, "end"] - df.loc[:, "start"]
    for c in list(df.columns):
        print(c)

    f1, _ = plot_prosody(df, plot=False)
    f2, _ = plot_diff(df, plot=False)
    plt.show()

    z = 4
    print(f"filler cross at {z}: ", (df["filler_now_cross"] == z).sum())
    print(f"omit cross at {z}: ", (df["omit_now_cross"] == z).sum())
