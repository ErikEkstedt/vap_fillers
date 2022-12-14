from os.path import join
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival

from vap_fillers.filler_model_process import find_cross
from vap_fillers.main import load_fillers
from vap_fillers.plot_utils import plot_mel_spectrogram, plot_vad, plot_speaker_probs
from vap_fillers.utils import load_model, pad_silence
from vap.utils import read_json, read_txt
from vap.audio import load_waveform

rel_paths = read_json("data/relative_audio_path.json")
AUDIO_ROOT = "../../data/switchboard/audio"
DA_UTT = "data/swb_dialog_acts_utterances"
TEST_PATH = "data/splits/test.txt"


"""

* grounded in 'clear' shifts
    - in dialog acts (questions?)
    - Model out over entire sessions? shift where there should be a shift.
"""


def load_question_audio(question, context=20, sample_rate=16_000):
    start = question.end - context
    if start < 0:
        start = 0
        context = question.end

    audio_path = join(AUDIO_ROOT, rel_paths[str(question.session)] + ".wav")
    y, _ = load_waveform(
        audio_path,
        start_time=start,
        end_time=question.end,
        sample_rate=sample_rate,
    )
    y = y.unsqueeze(0)  # add batch dim
    return y, context


def load_filler_audio(f, sample_rate=16_000, extract_prosody=False):
    speaker_idx = 0 if f.speaker == "A" else 1
    other_idx = 0 if speaker_idx == 1 else 1

    if extract_prosody:
        print("Extract prosody of filler")
        raise NotImplementedError("No prosody filler done")
    y, _ = load_waveform(
        join(AUDIO_ROOT, rel_paths[str(f.session)] + ".wav"),
        start_time=f.start,
        end_time=f.end,
        sample_rate=sample_rate,
    )
    y[other_idx].fill_(0.0)
    return y.unsqueeze(0)  # add batch dim


def cond_only_qw(utt):
    qs = [d for d in list(utt["da"]) if str(d) == "qw"]
    if len(qs) > 0:
        return True
    return False


def cond_only_qy(utt):
    qs = [d for d in list(utt["da"]) if str(d) == "qy"]
    if len(qs) > 0:
        return True
    return False


def cond_only_ends_with_qy(utt):
    qs = [d for d in list(utt["da"]) if str(d) == "qy"]
    if len(qs) > 0:
        if "qy" in utt.da[-2:]:
            return True
    return False


def extract_da(condition, sessions):
    questions = []
    for session in tqdm(sessions, desc="Extract all questions"):
        # sw2005A-utt-da.csv
        for speaker in ["A", "B"]:
            df = pd.read_json(join(DA_UTT, f"sw{session}{speaker}-utt-da.json"))
            for ii in range(len(df)):
                utt = df.iloc[ii]
                if condition(utt):
                    addition = pd.Series({"session": session, "speaker": speaker})
                    questions.append(pd.concat([addition, utt]))
    questions = pd.DataFrame(questions)
    return questions


def extract_valid_questions_with_fillers(
    model,
    qy,
    fillers,
    silence=10,
    min_filler_duration=0.2,
):
    def is_valid_turn_shift_check(out, speaker_idx, silence):
        sil_frames = round(silence * model.frame_hz)
        shift = out["p_now"][0, -sil_frames:, speaker_idx].cpu() <= 0.5
        cross_indices = torch.where(shift)[0]
        cross_idx = -1
        if len(cross_indices) == 0:
            # print('no cross')
            return False

        # Only shift after cross?
        cross_idx = cross_indices[0]
        back_to_current_speaker = torch.logical_not(shift)[cross_idx:].sum()
        if back_to_current_speaker > 0:
            # print('went back')
            return False

        return True

    def filler_exists(q, fillers):
        filler_cands = fillers[fillers["session"] == int(q.session)]

        if len(filler_cands) == 0:
            # print('no fillers at all')
            return False

        if not q.speaker in filler_cands["speaker"].unique():
            # print('no filler from speaker')
            return False

        speaker_cands = filler_cands[filler_cands["speaker"] == q.speaker]
        speaker_cands = speaker_cands[speaker_cands["duration"] >= min_filler_duration]

        if len(speaker_cands) == 0:
            # print('no valid speaker fillers (duration)')
            return False

        return True

    valid = []
    invalid = {"ts": 0, "fillers": 0}
    for ii in tqdm(range(len(qy)), desc="Find valid TS"):
        question = qy.iloc[ii]
        speaker_idx = 0 if question.speaker == "A" else 1

        if not filler_exists(question, fillers):
            invalid["fillers"] += 1
            continue

        waveform, _ = load_question_audio(question)
        x = pad_silence(waveform, silence=silence, sample_rate=model.sample_rate)
        out = model.probs(x.to(model.device))

        if not is_valid_turn_shift_check(out, speaker_idx, silence):
            invalid["ts"] += 1
            continue

        fill = fillers[fillers["session"] == int(question.session)]
        fill = fill[fill["speaker"] == question.speaker]
        fill = fill[fill["duration"] >= min_filler_duration]
        valid.append({"question": question, "filler": fill})
    for k, v in invalid.items():
        print(f"{k}: {v}")
    return valid


def extract_added_filler_cross(
    model,
    valid,
    silence=10,
    filler_pre_pad_frames=10,
    add_filler_pre_pause=False,
    save_plots=False,
    fig_root="results/images/questions_fillers",
):
    sil_frames = int(silence * model.frame_hz)
    end_frame = sil_frames - 1

    if save_plots:
        Path(fig_root).mkdir(parents=True, exist_ok=True)

    data = []

    try:
        for v in tqdm(valid, desc="Extract QY+filler cross"):
            q = v["question"]
            speaker_idx = 0 if q.speaker == "A" else 1

            waveform, rel_q_end = load_question_audio(q)
            rel_q_start = rel_q_end - (q.end - q.start)
            x = pad_silence(waveform, silence=silence, sample_rate=model.sample_rate)
            original_out = model.probs(x.to(model.device))

            # Extract crosses
            q_cross = find_cross(original_out["p_now"][0, -sil_frames:, speaker_idx])
            q["q_cross"] = q_cross
            q["q_is_filler"] = 0

            data.append(q)
            # data.append({"filler": 0, "cross": q_cross, "id": q.utt_idx, "n": 0})
            # Fillers
            for fi in range(len(v["filler"])):
                filler = v["filler"].iloc[fi].copy()
                wfill = load_filler_audio(filler)  # zeros out other speaker
                if add_filler_pre_pause:
                    pad = torch.zeros((1, 2, filler_pre_pad_frames))
                    wfill = torch.cat((pad, wfill), dim=-1)

                # Add filler directly after 'qy'
                wfill = torch.cat((waveform, wfill), dim=-1)
                x = pad_silence(wfill, silence=silence, sample_rate=model.sample_rate)

                out = model.probs(x.to(model.device))
                f_cross = find_cross(out["p_now"][0, -sil_frames:, speaker_idx])

                filler["q_idx"] = q.utt_idx
                filler["q_cross"] = f_cross
                filler["q_is_filler"] = 1
                data.append(filler)

                if save_plots:
                    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(12, 6))
                    plot_question(
                        x, out, speaker_idx, rel_q_start, rel_q_end, ax=ax, plot=False
                    )
                    x_time = (
                        torch.arange(original_out["p_now"].shape[1]) / model.frame_hz
                    )
                    plot_speaker_probs(
                        x_time,
                        original_out["p_now"][0, :, speaker_idx].cpu(),
                        ax=ax[-1],
                    )
                    [a.set_xlim([0, x_time[-1]]) for a in ax]
                    fig.savefig(join(fig_root, f"{q.utt_idx}_F_{filler.utt_idx}.png"))
                    plt.close("all")
    except KeyboardInterrupt:
        print("Interrupted by keyboard...")

    df = pd.DataFrame(data)

    # Add survival/status
    df["status"] = df["q_cross"] >= 0
    df["survival"] = df["q_cross"]
    df["survival"][df["survival"] == -1] = end_frame
    return df


def survival_fillers(df, plot=False):
    # Different groups to plot in kaplan_meier
    Q = df[df["q_is_filler"] == 0]
    F = df[df["q_is_filler"] == 1]

    # Get structured array for log-rank test
    ll = []
    for st, su in zip(df["status"].values, df["survival"].values):
        ll.append((st, su))
    y = np.array(ll, dtype=[("name", "?"), ("age", "i8")])
    group = df["q_is_filler"].values

    # Survival test
    tq, pq = kaplan_meier_estimator(Q["status"], Q["survival"])
    tf, pf = kaplan_meier_estimator(F["status"], F["survival"])
    chisq, pvalue, stats, covariance = compare_survival(y, group, return_stats=True)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        fig.suptitle(f"Question (+ Fillers)\nChisq={round(chisq, 1)}, P={pvalue}")
        ax.step(tq / 50, pq, where="post", color="r", label="Question")
        ax.step(tf / 50, pf, where="post", color="teal", label="Question W/ Filler")
        ax.set_ylabel("est. probability of shift")
        ax.set_xlabel("time s")
        ax.legend()
        plt.tight_layout()
        plt.show()
    return {
        "tq": tq,
        "pq": pq,
        "tf": tf,
        "pf": pf,
        "chisq": chisq,
        "pvalue": pvalue,
        "stats": stats,
        "covariance": covariance,
    }


def plot_question(
    x,
    out,
    speaker_idx,
    rel_q_start,
    rel_q_end,
    frame_hz=50,
    sample_rate=16_000,
    ax=None,
    plot=True,
):
    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    else:
        fig = None
    other = 0 if speaker_idx == 1 else 1
    x_time = torch.arange(out["p_now"].shape[1]) / frame_hz
    if speaker_idx == 0:
        plot_mel_spectrogram(x[0].cpu(), ax=[ax[0], ax[1]], sample_rate=sample_rate)
        plot_vad(out["vad"][0, :, speaker_idx].cpu(), ax=ax[0])
        plot_vad(out["vad"][0, :, other].cpu(), ax=ax[1])
    else:
        plot_mel_spectrogram(x[0].cpu(), ax=[ax[1], ax[0]], sample_rate=sample_rate)
        plot_vad(out["vad"][0, :, speaker_idx].cpu(), ax=ax[0])
        plot_vad(out["vad"][0, :, other].cpu(), ax=ax[1])
    plot_speaker_probs(x_time, out["p_now"][0, :, speaker_idx].cpu(), ax=ax[2])
    ax[0].axvline(rel_q_end, color="r", linewidth=2)
    ax[1].axvline(rel_q_end, color="r", linewidth=2)
    ax[2].axvline(rel_q_end, color="r", linewidth=2)
    ax[0].axvline(rel_q_start, color="r", linewidth=2)
    ax[1].axvline(rel_q_start, color="r", linewidth=2)
    ax[2].axvline(rel_q_start, color="r", linewidth=2)
    ax[2].set_xlim([0, x_time[-1]])
    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig


if __name__ == "__main__":

    import matplotlib as mpl

    mpl.use("agg")

    # test_sessions = read_txt(TEST_PATH)
    model = load_model()

    fillers = load_fillers("results/all_fillers_test_prosody_model_da.csv")

    #####################################
    # Extract valid questions + fillers
    #####################################
    # qy = extract_da(cond_only_ends_with_qy, test_sessions)
    # valid = extract_valid_questions_with_fillers(
    #     model, qy, fillers, silence=5, min_filler_duration=0.4
    # )
    # print("Valid: ", len(valid))
    # torch.save(valid, 'valid.pt')
    valid = torch.load("valid.pt")

    #####################################
    # Extract model output cross
    #####################################
    df = extract_added_filler_cross(
        model,
        valid,
        silence=10,
        filler_pre_pad_frames=10,
        add_filler_pre_pause=False,
        save_plots=True,
    )

    # df.to_csv("results/questions_and_fillers.csv")
    # df = pd.read_csv("results/questions_and_fillers.csv")

    # result = survival_fillers(df, plot=True)
