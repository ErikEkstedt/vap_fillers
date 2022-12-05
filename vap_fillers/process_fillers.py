from argparse import ArgumentParser
from os.path import expanduser, join
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torchaudio

from vap.audio import load_waveform
from vap.functional import pitch_praat, intensity_praat
from vap.utils import read_json, read_txt, write_txt, write_json, tensor_dict_to_json

from vap_fillers.utils import load_model, pad_silence
from vap_fillers.plot_utils import plot_mel_spectrogram, plot_speaker_probs
from vap_fillers.dialog_act_filler_matching import find_filler_pos_rel_da

"""
Load the model and extract how it thinks dialogs will proceed after a filler
and when omitted the filler. 


1. Combine all fillers (uh/um) into single data structure
2. Extract model continuation after (omitted) fillers
3. Extract the prosody of the filler
4. Extract model output + waveform and save to show videos.


Saves to csv through pandas
COLUMNS = [
    "filler_id",
    "session",
    "start",
    "end",
    "speaker",
    "uh_or_um",
    "f0",
    "f0_global_m",
    "f0_global_s",
    "intensity",
    "intensity_global_m",
    "intensity_global_s",
    "da",
    "da_loc",
    "da_n_words",
    "filler_now_cross",
    "filler_fut_cross",
    "omit_now_cross",
    "omit_fut_cross",
]

"""

# Kept for reference
TEST_FILE_PATH_UH = "data/uh_test.txt"
TEST_FILE_PATH_UM = "data/um_test.txt"

# DEFAULT PATHS
TEST_FILLER_PATH = "data/test_fillers.txt"
OUTPUT_PATH = "results/fillers"

# Data
REL_PATH = "data/relative_audio_path.json"
AUDIO_ROOT = join(expanduser("~"), "projects/data/switchboard/audio")
REL_PATHS = read_json(REL_PATH)

# Prosody
SESSION_INTENS_PATH = "data/intensity_information_test.json"
SESSION_F0_PATH = "data/pitch_information_test.json"
SESSION_INTENS = read_json(SESSION_INTENS_PATH)
SESSION_F0 = read_json(SESSION_F0_PATH)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-f", "--fillers", type=str, default=TEST_FILLER_PATH)
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="../VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt",
        help="Path to trained model",
    )
    parser.add_argument(
        "-r",
        "--audio_root",
        type=str,
        default=AUDIO_ROOT,
        help="Path to swb audio",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=OUTPUT_PATH,
        help="filename to save data to",
    )
    parser.add_argument(
        "--context",
        type=float,
        default=20,
        help="Duration of context prior to filler/no-filler",
    )
    parser.add_argument(
        "--silence",
        type=float,
        default=10,
        help="Duration of silence after the fillers/no-filler",
    )
    parser.add_argument("--save_figs", action="store_true")
    parser.add_argument("--save_model_out", action="store_true")
    args = parser.parse_args()

    print("-" * 50)
    print("ARGUMENTS")
    print("---------")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-" * 50)
    return args


def create_paths(args):
    path = args.output
    Path(path).mkdir(parents=True, exist_ok=True)
    model_path = join(path, "model")
    figure_path = join(path, "figures")

    if args.save_model_out:
        Path(model_path).mkdir(parents=True, exist_ok=True)

    if args.save_figs:
        Path(figure_path).mkdir(parents=True, exist_ok=True)
    return model_path, figure_path


# This is only kept for reference. See result in `TEST_FILLER_PATH`
def combine_fillers():
    fillers = []
    uh = read_txt(TEST_FILE_PATH_UH)
    filler_type = "uh"
    for filler in uh:
        session, start, end, speaker = filler.split()
        fillers.append(f"{session} {start} {end} {speaker} {filler_type}")

    um = read_txt(TEST_FILE_PATH_UM)
    filler_type = "um"
    for filler in um:
        session, start, end, speaker = filler.split()
        fillers.append(f"{session} {start} {end} {speaker} {filler_type}")

    write_txt(fillers, TEST_FILLER_PATH)
    print("Saved all fillers -> ", TEST_FILLER_PATH)
    return fillers


def extract_prosody(y, session, speaker, sample_rate=16_000, hop_time=0.01):
    speaker = "A" if speaker == 0 else "B"

    p = pitch_praat(y, hop_time=hop_time, sample_rate=sample_rate)
    pm = SESSION_F0[session][speaker]["mean"]
    ps = SESSION_F0[session][speaker]["std"]
    p = (p - pm) / ps

    i = intensity_praat(y, hop_time=hop_time, sample_rate=sample_rate)
    im = SESSION_INTENS[session][speaker]["mean"]
    ist = SESSION_INTENS[session][speaker]["std"]
    i = (i - im) / ist

    return {
        "f0": {
            "filler": round(p.mean().item(), 3),
            "mean": round(pm, 3),
            "std": round(ps, 3),
        },
        "intensity": {
            "filler": round(i.mean().item(), 3),
            "mean": round(im, 3),
            "std": round(ist, 3),
        },
    }


def find_shift_cross(out, speaker, start_frame, cutoff=0.5):
    """
    Finds the first frame in "p_now" or "p_future" that crosses over the 50% cutoff
    """

    # all p-frames after filler/no-filler
    pn = out["p_now"][0, start_frame:, speaker]
    now_shift_idx = torch.where(pn <= cutoff)[0]

    if len(now_shift_idx) < 1:
        now_shift_idx = -1
    else:
        now_shift_idx = now_shift_idx[0].item()

    pf = out["p_future"][0, start_frame:, speaker]
    fut_shift_idx = torch.where(pf <= cutoff)[0]
    if len(fut_shift_idx) < 1:
        fut_shift_idx = -1
    else:
        fut_shift_idx = fut_shift_idx[0].item()

    return now_shift_idx, fut_shift_idx


def plot_filler(y, out, speaker, title="", sample_rate=16_000, frame_hz=50):
    n_frames = out["p_now"].shape[1]
    x = torch.arange(n_frames) / frame_hz

    fig, ax = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(title)
    plot_mel_spectrogram(y[0].cpu(), ax=[ax[0], ax[1]], sample_rate=sample_rate)
    plot_speaker_probs(x, out["p_now"][0, :, speaker].cpu(), ax=ax[2], label="P-now")
    ax[2].set_yticks([-0.25, 0.25])
    ax[2].set_yticklabels(["B", "A"])
    plot_speaker_probs(
        x, out["p_future"][0, :, speaker].cpu(), ax=ax[3], label="P-future"
    )
    ax[3].set_yticks([-0.25, 0.25])
    ax[3].set_yticklabels(["B", "A"])
    plt.subplots_adjust(
        left=0.05, bottom=None, right=0.99, top=0.95, wspace=0.01, hspace=0.03
    )
    return fig


def extract_filler_out_and_prosody(
    waveform,
    model,
    session,
    speaker,
    silence,
    filler_duration,
    filler_duration_samples,
    context_samples,
    name="",
    save_figs=False,
    fig_path="figure",
    save_model_out=False,
    model_path="model_out",
):
    """
    Returns:

    {
        "now_cross": now_cross,
        "fut_cross": fut_cross,
        "f0": filler_f0,
        "intensity": filler_int,
    }
    """

    ############################################################
    # PROSODY
    ############################################################
    if filler_duration >= 0.11:
        only_filler_wav = waveform[
            ..., context_samples : context_samples + filler_duration_samples + 1
        ]
    else:
        tmp_end = context_samples + filler_duration_samples + 1
        tmp_start = tmp_end - int(0.11 * model.sample_rate)
        only_filler_wav = waveform[..., tmp_start:tmp_end]

    prosody = extract_prosody(
        only_filler_wav, session, speaker, sample_rate=model.sample_rate
    )

    ############################################################
    # MODEL OUT
    ############################################################
    wav_filler = pad_silence(waveform, silence=silence, sample_rate=model.sample_rate)
    out_filler = model.probs(wav_filler.to(model.device))

    silence_frames = int(silence * model.frame_hz)
    sil_start_frame = out_filler["p_now"].shape[1] - silence_frames
    now_cross, fut_cross = find_shift_cross(
        out_filler, speaker, start_frame=sil_start_frame
    )

    # Save figure
    if save_figs:
        fig = plot_filler(wav_filler, out_filler, speaker, title=f"Filler: {name}")
        fig.savefig(join(fig_path, name + ".png"))
        plt.close("all")

    if save_model_out:
        # Save model output
        write_json(
            tensor_dict_to_json(out_filler), join(args.model_path, name + ".json")
        )
        torchaudio.save(
            join(model_path, name + ".wav"),
            wav_filler[0].cpu(),
            sample_rate=model.sample_rate,
        )

    # w_or_wo = 1
    # with_filler_row = f"{session} {filler_id} {s} {e} {speaker} {uh_or_um} {w_or_wo} {now_cross} {fut_cross} {filler_f0} {filler_int}"
    return {
        "now_cross": now_cross,
        "fut_cross": fut_cross,
        "f0": prosody["f0"],
        "intensity": prosody["intensity"],
    }


def extract_non_filler_out(
    waveform,
    model,
    speaker,
    silence_frames,
    filler_start,
    filler_end,
    name="",
    save_figs=False,
    fig_path="figure",
    save_model_out=False,
    model_path="model_out",
):
    """ """

    ############################################################
    # Omit filler
    ############################################################
    fill_dur = filler_end - filler_start
    fill_n_samples = int(fill_dur * model.sample_rate)
    wav_omit = waveform[..., :-fill_n_samples]
    wav_omit = pad_silence(
        wav_omit, silence=args.silence, sample_rate=model.sample_rate
    )
    out_omit = model.probs(wav_omit.to(model.device))

    sil_start_frame = out_omit["p_now"].shape[1] - silence_frames

    now_cross, fut_cross = find_shift_cross(
        out_omit, speaker, start_frame=sil_start_frame
    )

    # Save figure
    if save_figs:
        fig = plot_filler(wav_omit, out_omit, speaker, title=f"Omitted: {name}")
        fig.savefig(join(fig_path, name + "_omit.png"))
        plt.close("all")

    # Save model output
    if save_model_out:
        write_json(
            tensor_dict_to_json(out_omit), join(args.model_path, name + "_omit.json")
        )
        torchaudio.save(
            join(model_path, name + "_omit.wav"),
            wav_omit[0].cpu(),
            sample_rate=model.sample_rate,
        )

    return {"now_cross": now_cross, "fut_cross": fut_cross}


def extract_filler_segment(filler_id, filler_row, model, args):
    """
    ROW ENTRIES:

    "SESSION S E SPEAKER UH_OR_UM WITH_OR_OMIT_FILLER NOW_CROSS FUT_CROSS FILLER_F0 FILLER_INT"
    """

    # Get information from row
    session, s, e, speaker, uh_or_um = filler_row.split()
    speaker = 0 if speaker == "A" else 1
    filler_start = float(s)
    filler_end = float(e)
    context_start = filler_start - args.context
    name = f"{session}-{str(filler_id).zfill(4)}-{speaker}-{uh_or_um}"

    if context_start < 0:
        return None
        # context_start = 0

    # Relative frames in output
    silence_frames = int(args.silence * model.frame_hz)
    context_samples = args.context * model.sample_rate

    filler_duration = filler_end - filler_start
    filler_duration_samples = int(filler_duration * model.sample_rate)

    ############################################################
    # Load waveform, add batch dim and move to correct device
    # add batch dimension -> (B, 2, n_samples)
    ############################################################
    audio_path = join(args.audio_root, REL_PATHS[session] + ".wav")
    waveform, _ = load_waveform(
        audio_path, start_time=context_start, end_time=filler_end
    )
    waveform = waveform.unsqueeze(0)

    ############################################################
    # Include filler
    ############################################################
    filler_info = extract_filler_out_and_prosody(
        waveform,
        model,
        session=session,
        speaker=speaker,
        silence=args.silence,
        filler_duration=filler_duration,
        filler_duration_samples=filler_duration_samples,
        context_samples=context_samples,
        name=name,
        save_figs=args.save_figs,
        fig_path=args.fig_path,
        save_model_out=args.save_model_out,
        model_path=args.model_path,
    )

    ############################################################
    # Omit filler
    ############################################################
    no_filler_info = extract_non_filler_out(
        waveform,
        model,
        speaker=speaker,
        silence_frames=silence_frames,
        filler_start=filler_start,
        filler_end=filler_end,
        name=name,
        save_figs=args.save_figs,
        fig_path=args.fig_path,
        save_model_out=args.save_model_out,
        model_path=args.model_path,
    )

    ############################################################
    # Dialog Act position
    ############################################################
    da_loc, da_n_words, da = find_filler_pos_rel_da(
        session, speaker, filler_start, filler_end
    )

    return {
        "filler_id": filler_id,
        "session": session,
        "start": s,
        "end": e,
        "speaker": speaker,
        "uh_or_um": uh_or_um,
        "f0": filler_info["f0"]["filler"],
        "f0_global_m": filler_info["f0"]["mean"],
        "f0_global_s": filler_info["f0"]["std"],
        "intensity": filler_info["intensity"]["filler"],
        "intensity_global_m": filler_info["intensity"]["mean"],
        "intensity_global_s": filler_info["intensity"]["std"],
        "da": da,
        "da_loc": da_loc,
        "da_n_words": da_n_words,
        "filler_now_cross": filler_info["now_cross"],
        "filler_fut_cross": filler_info["fut_cross"],
        "omit_now_cross": no_filler_info["now_cross"],
        "omit_fut_cross": no_filler_info["fut_cross"],
    }


if __name__ == "__main__":

    args = get_args()
    args.model_path, args.fig_path = create_paths(args)
    print("-" * 50)
    print("PATHS")
    print("-----")
    print("output: ", args.output)
    print("model: ", args.model_path)
    print("figure: ", args.fig_path)
    print("-" * 50)

    fillers = read_txt(args.fillers)
    model = load_model(args.checkpoint)

    #######################################################################

    all_filler_data = []
    processed = 0
    try:
        for filler_id, filler_row in enumerate(tqdm(fillers, desc="Process fillers")):
            row_dict = extract_filler_segment(filler_id, filler_row, model, args)
            if row_dict is not None:
                all_filler_data.append(row_dict)
            processed += 1
    except KeyboardInterrupt:
        print("ABORTED")
    skipped = processed - len(all_filler_data)
    print(f"Skipped: {skipped} (by context)")
    df = pd.DataFrame(all_filler_data)
    datapath = join(args.output, "filler_info.csv")
    df.to_csv(datapath, index=False)
    print("Saved CSV -> ", datapath)

    # print(all_filler_data)
    # if isinstance(skip_or_rows, str):
    #     skipped[skip_or_rows] += 1
    #     continue
    # result_file.write(skip_or_rows[0] + "\n")
    # result_file.write(skip_or_rows[1] + "\n")
    # processed += 1
    # result_file.close()
    #
    # print(f"Processed {processed} fillers")
    #
    # total_skips = sum([v for _, v in skipped.items()])
    # print("-" * 50)
    # print(f"SKIPPED: {total_skips}")
    # for k, v in skipped.items():
    #     print(f"{k}: {v}")
    # print("-" * 50)
