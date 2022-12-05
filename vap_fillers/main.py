import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

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


if __name__ == "__main__":

    df = pd.read_csv("data/all_filler_info.csv")

    filler_cross = df.loc[df["NOW_CROSS"] > 0]

    f = df.loc[df["NOW_CROSS"] == -1]
    f = f.loc[f["DIFF_NOW_CROSS"] != 0]

    df = pd.read_csv(
        "data/test_fillers_dal.txt",
        # names = ["SESSION", "START", "END", "SPEAKER", "UH_OR_UM", "LOC", "WORDS_IN_DA", "DA"]
        names=["SESSION", "S", "E", "SPEAKER", "UH_OR_UM", "LOC", "WORDS_IN_DA", "DA"],
    )
    df = df.sort_values("SESSION")
    df2 = pd.read_csv("results/fillers/filler_output.txt", sep=" ")
    df2 = df2.sort_values("SESSION")

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

    # Add 2 columns for cross-differences
    # Add 3 columns for "LOC", "WORDS_IN_DA", "DA"

    # Combine the information by looking at start time

    df = pd.read_csv("data/all_filler_info.csv")

    # fillers = np.array([f.split(",") for f in read_txt("data/test_fillers_dal.txt")])
    # fig, _ = plot_filler_da_loc_distribution(fillers)
    # plt.show()

    d = read_txt("results/fillers/filler_output.txt")
    fillers = np.array([f.split() for f in d])
    header = list(fillers[0])
    fillers = fillers[1:]

    # with pandas

    # 1. Add a column with the difference of NOW: with_fill - omit_fill, (FUT: with_fill - omit_fill)
    # 2. Remove all rows where cross=-1 (don't terminate)
    #       - Keep all fillers which made a difference
    # 3. How to handle the case where the filler cross=-1 and no_fill 0 < cross<= 500
    # 4. Find filler-id where the differences are large
    # 5. Find filler-id where the differences are large and cross happens < 200 (fast)

    # Cross
    # Remove where both filler/nofill has p_cross=-1, neither terminates
    #

    df["NOW_CROSS"].mean()

    uh_df = df.loc[lambda x: x["UH_OR_UM"] == "uh"]
    with_uh_cross = uh_df.loc[lambda x: x["WITH_OR_OMIT_FILLER"] == 1][
        "NOW_CROSS"
    ].mean()
    omit_uh_cross = uh_df.loc[lambda x: x["WITH_OR_OMIT_FILLER"] == 0][
        "NOW_CROSS"
    ].mean()
    print("UH")
    print("UH: ", with_uh_cross)
    print("omit: ", omit_uh_cross)

    um_df = df.loc[lambda x: x["UH_OR_UM"] == "um"]
    with_um_cross = um_df.loc[lambda x: x["WITH_OR_OMIT_FILLER"] == 1][
        "NOW_CROSS"
    ].mean()
    omit_um_cross = um_df.loc[lambda x: x["WITH_OR_OMIT_FILLER"] == 0][
        "NOW_CROSS"
    ].mean()
    print("UM")
    print("um: ", with_um_cross)
    print("omit: ", omit_um_cross)

    df["NOW_CROSS"].hist()
    plt.show()

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
