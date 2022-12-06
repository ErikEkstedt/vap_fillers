import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import expanduser, join

from vap.audio import load_waveform
from vap.utils import read_json

from vap_fillers.plot_utils import plot_mel_spectrogram, plot_speaker_probs
from vap_fillers.utils import load_model, pad_silence, moving_average

# Data
REL_PATH = "data/relative_audio_path.json"
AUDIO_ROOT = join(expanduser("~"), "projects/data/switchboard/audio")
REL_PATHS = read_json(REL_PATH)


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


def plot_diff_location_box(df, beg_lim=0.33, end_lim=0.66, plot=True):
    y = df["diff"]
    x = df["da_loc"] / df["da_n_words"]
    x[x <= beg_lim] = 0
    x[x >= end_lim] = 2
    m = np.logical_and(beg_lim < x, x < end_lim)
    x[m] = 1
    beg = np.where(x == 0)[0]
    mid = np.where(x == 1)[0]
    end = np.where(x == 2)[0]

    # xb = x[beg]
    yb = y[beg]
    ym = y[mid]
    ye = y[end]

    fig, ax = plt.subplots(1, 1)
    ax.boxplot([yb.values, ym.values, ye.values], showfliers=False)
    ax.set_xticklabels(["beginning", "mid", "end"])
    ax.set_ylabel("Diff (Filler-Omitted)")
    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig, ax


def omit_mutual_survival(df):
    permiss = np.logical_and(df["filler_now_cross"] != -1, df["omit_now_cross"] != -1)
    return df[permiss]


def find_good_fillers(df, diff_thresh=50, cross_thresh=100, min_dur=0):
    # diff larger > 1s
    large_diff = df[df["diff"] >= diff_thresh]
    # Early Cross
    gf = large_diff[large_diff["omit_now_cross"] <= cross_thresh]
    gf = gf[gf["omit_now_cross"] >= 0]
    if min_dur > 0:
        gf = gf[gf["duration"] >= min_dur]
    return gf


def find_difference(df, direction="pos", relative=True, verbose=False):
    assert direction in [
        "pos",
        "neg",
        "zero",
    ], f"Bad direction choose: ['pos', 'neg', 'zero'] got {direction}"

    if verbose:
        print("-" * 30)
        print(f"Diff {direction}")
        print("Total -> ", len(df))
    d = omit_mutual_survival(df)
    if verbose:
        print("At least 1 cross -> ", len(d))
    d = d[d["omit_now_cross"] != -1]
    if verbose:
        print("Omit crosses -> ", len(d))

    if relative:
        if direction == "pos":
            d = d[d["diff"] > 0]
        elif direction == "neg":
            d = d[d["diff"] < 0]
        else:
            d = d[d["diff"] == 0]
    else:
        abs_diff = d["duration"] + d["diff"]
        if direction == "pos":
            d = d[abs_diff > 0]
        elif direction == "neg":
            d = d[abs_diff < 0]
        else:
            d = d[abs_diff == 0]

    if verbose:
        if relative:
            print(f"Relative {direction.upper()} diff -> ", len(d))
        else:
            print(f"Absolute {direction.upper()} diff -> ", len(d))
        print("-" * 30)
    return d


def find_reverse_fillers(df, diff_thresh=20, min_dur=0):
    rf = df[df["diff"] < -diff_thresh]
    rf = rf[rf["filler_now_cross"] >= 0]
    if min_dur > 0:
        rf = rf[rf["duration"] >= min_dur]
    return rf


def find_no_cross(df, min_dur=0):
    no_cross = np.logical_and(df["filler_now_cross"] == -1, df["omit_now_cross"] == -1)
    nd = df[no_cross]
    if min_dur > 0:
        nd = nd[nd["duration"] >= min_dur]
    return nd


# def valid_filler(df):
#     # Only keep omitted that did not survive
#     d = df[df["omit_now_cross"] != -1]


def extract_and_plot_diff_bars(df, min_dur=0, plot=True):
    if min_dur > 0:
        d = df[df["duration"] >= min_dur]
    else:
        d = df
    data = {"tot": len(d)}
    relative = False
    for rel in [True, False]:
        name = "relative" if rel else "absolute"
        data[name] = {}
        for direction in ["pos", "neg", "zero"]:
            data[name][direction] = len(
                find_difference(d, relative=rel, direction=direction)
            )
        data[name]["no_cross"] = len(find_no_cross(d))

    ########
    # PLOT #
    ########
    y_rel = 100 * np.array(list(data["relative"].values())) / data["tot"]
    y_abs = 100 * np.array(list(data["absolute"].values())) / data["tot"]
    xlabels = list(data["relative"].keys())
    x = np.arange(len(y_rel))

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f"Diff (duration > {min_dur}s) Total: {data['tot']}")
    w = 0.4
    ax.bar(x - w / 2, y_rel, width=w, label="Relative")
    ax.bar(x + w / 2, y_abs, width=w, label="Absolute")
    ax.legend()
    # ax.set_xticklabels(["Positive", "Negative", "Zero", "No-cross"])
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("%")
    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig, data


def load_fillers(path="results/filler_info/filler_info.csv"):
    df = pd.read_csv(path)
    df["diff"] = df.loc[:, "filler_now_cross"] - df.loc[:, "omit_now_cross"]
    df["duration"] = df.loc[:, "end"] - df.loc[:, "start"]
    # for c in list(df.columns):
    #     print(c)
    return df


def load_filler(filler, context=20, silence=10, sample_rate=16_000):
    audio_start = filler["start"] - context

    if audio_start < 0:
        audio_start = 0
        context = filler["start"]

    audio_end = filler["end"]

    audio_path = join(AUDIO_ROOT, REL_PATHS[str(filler["session"])] + ".wav")
    waveform, _ = load_waveform(
        audio_path,
        start_time=audio_start,
        end_time=audio_end,
        sample_rate=sample_rate,
    )
    waveform = waveform.unsqueeze(0)  # add batch dim

    w_filler = pad_silence(waveform, silence=silence, sample_rate=sample_rate)

    # Remove filler
    fill_n_samples = int(filler["duration"] * sample_rate)
    w_omit = waveform[..., :-fill_n_samples]

    diff_samples = w_filler.shape[-1] - w_omit.shape[-1]

    # Add silences
    w_omit = pad_silence(w_omit, sil_samples=diff_samples)

    # Batch
    return torch.cat([w_filler, w_omit]), context


def plot_vad(vad, ax, frame_hz=50):
    assert vad.ndim == 1, f"Expects (N_FRAMES, ) got {vad.shape}"
    ymin, ymax = ax.get_ylim()
    scale = ymax - ymin
    x = torch.arange(len(vad)) / frame_hz
    ax.plot(x, ymin + vad.cpu() * scale, color="w")


def plot_filler(
    y,
    out,
    speaker,
    rel_filler_start=None,
    filler_dur=None,
    title="",
    sample_rate=16_000,
    frame_hz=50,
    smooth=0,
):
    n_frames = out["p_now"].shape[1]
    x = torch.arange(n_frames) / frame_hz
    fig, ax = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(title)

    other = 0 if speaker == 1 else 1

    if speaker == 0:
        plot_mel_spectrogram(y[0].cpu(), ax=[ax[0], ax[1]], sample_rate=sample_rate)
        plot_vad(out["vad"][0, :, speaker], ax=ax[0])
        plot_vad(out["vad"][0, :, other], ax=ax[1])
    else:
        plot_mel_spectrogram(y[0].cpu(), ax=[ax[1], ax[0]], sample_rate=sample_rate)
        plot_vad(out["vad"][0, :, speaker], ax=ax[0])
        plot_vad(out["vad"][0, :, other], ax=ax[1])

    y = out["p_now"][..., speaker].cpu()
    if smooth > 0:
        y = moving_average(y, k=smooth)
    plot_speaker_probs(x, y[0].cpu(), ax=ax[2], label="P-now FILLER")
    plot_speaker_probs(
        x,
        y[1],
        ax=ax[3],
        label="P-now OMIT",
    )
    ax[2].set_yticks([-0.25, 0.25])
    ax[2].set_yticklabels(["B", "A"])
    ax[3].set_yticks([-0.25, 0.25])
    ax[3].set_yticklabels(["B", "A"])

    if rel_filler_start is not None:
        ax[0].axvline(rel_filler_start, color="r")
        ax[1].axvline(rel_filler_start, color="r")
        ax[2].axvline(rel_filler_start, color="r")
        ax[3].axvline(rel_filler_start, color="r")

        if filler_dur is not None:
            ax[2].axvline(rel_filler_start + filler_dur, color="r", linestyle="dashed")
            ax[3].axvline(rel_filler_start + filler_dur, color="r", linestyle="dashed")

    for a in ax[2:]:
        a.legend(loc="upper right")

    plt.subplots_adjust(
        left=0.05, bottom=None, right=0.99, top=0.95, wspace=0.01, hspace=0.03
    )
    return fig


def plot_loop(df):
    for ii in range(len(df)):
        filler = df.iloc[ii]
        x, rel_filler_start = load_filler(filler)
        out = model.probs(x.to(model.device))
        fig = plot_filler(
            x,
            out,
            speaker=filler["speaker"],
            rel_filler_start=rel_filler_start,
            smooth=0,
        )
        try:
            plt.show()
        except KeyboardInterrupt:
            plt.close("all")
            break


if __name__ == "__main__":

    # f1, _ = plot_prosody(df, plot=False)
    # f2, _ = plot_diff(df, plot=False)
    f3, _ = plot_diff_location_box(df, plot=True)
    # plt.show()

    df = load_fillers()
    model = load_model()

    fig, data = extract_and_plot_diff_bars(df, min_dur=0)

    pf = find_difference(df, direction="pos", relative=True)
    nf = find_difference(df, direction="neg", relative=False)
    zf = find_difference(df, direction="zero", relative=True)

    plot_loop(pf)
