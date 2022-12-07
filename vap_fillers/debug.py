import pandas as pd
import matplotlib.pyplot as plt


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


def weird_intensity():
    import numpy as np
    from glob import glob
    from os.path import basename, join

    from datasets_turntaking.dialog_audio_dataset import (
        load_spoken_dialog_audio_dataset,
    )

    ANNO_PATH = "../../data/switchboard/annotations/swb_ms98_transcriptions"

    all_sessions = glob(join(ANNO_PATH, "**/*A-ms98-a-trans.text"), recursive=True)
    all_sessions = [
        basename(s).split("-")[0].replace("sw", "").replace("A", "").replace("B", "")
        for s in all_sessions
    ]
    prosody = []

    session = "2167"
    speaker = "B"

    # try:
    dset = load_spoken_dialog_audio_dataset(["switchboard"], split="train")
    session_indices = np.array(dset["session"])

    session in list(session_indices)

    for session in all_sessions:

        if session not in session_indices:
            continue

        if session in prosody:
            continue

        # Get data sample
        idx = np.where(session_indices == session)[0].item()
        d = dset[idx]

        # extract prosody
        tmp_prosody = self.extract_prosody_from_session(d, session)
        prosody.append(tmp_prosody)


if __name__ == "__main__":

    df = pd.read_csv("data/FILLER/all_fillers_test_prosody.csv")

    pdf = pd.read_csv("data/FILLER/swb_prosody.csv")
    for c in df.columns:
        print(c)
