import torch
import numpy as np
from tqdm import tqdm
from os.path import expanduser, join
import matplotlib.pyplot as plt


from datasets_turntaking.dialog_audio_dataset import load_spoken_dialog_audio_dataset
from vap.audio import load_waveform, get_audio_info
from vap.functional import pitch_praat
from vap.utils import read_txt, read_json, write_txt, write_json


UH_ALL_PATH = "data/uh.txt"
UM_ALL_PATH = "data/um.txt"
UH_PATH = "data/uh_test.txt"
UM_PATH = "data/um_test.txt"
REL_PATH = "data/relative_audio_path.json"
TEST_FILE_PATH = "data/test.txt"
AUDIO_ROOT = join(expanduser("~"), "projects/data/switchboard/audio")
SAMPLE_RATE = 16_000


def filter_by_test_split(path):
    fillers = read_txt(path)
    ok_files = read_txt(TEST_FILE_PATH)

    filtered_fillers = []

    skipped_by_file = 0
    for filler in fillers:
        session, fill_start, fill_end, speaker = filler.split()
        if session not in ok_files:
            skipped_by_file += 1
            continue
        filtered_fillers.append(filler)

    print("Skipped: ", skipped_by_file)
    return filtered_fillers


def _create_test_filler_split():
    print("UH")
    uh_test_fillers = filter_by_test_split(UH_ALL_PATH)
    print("uh test: ", len(uh_test_fillers))
    print("UM")
    um_test_fillers = filter_by_test_split(UM_ALL_PATH)
    print("um test: ", len(um_test_fillers))
    write_txt(uh_test_fillers, UH_PATH)
    write_txt(um_test_fillers, UM_PATH)


def _extract_pitch(savepath="data/pitch_information_test.json"):
    session_to_rel_path = read_json(REL_PATH)
    test_sessions = read_txt(TEST_FILE_PATH)
    dset = load_spoken_dialog_audio_dataset(
        ["switchboard"], split="test", min_word_vad_diff=0.1
    )
    sess_idx = np.array(dset["session"])

    def extract_pitch_from_session(session, pitch, min_chunk_time=0.1):
        min_samples = min_chunk_time * SAMPLE_RATE

        idx = np.where(sess_idx == session)[0].item()
        audio_path = join(AUDIO_ROOT, session_to_rel_path[session] + ".wav")
        d = dset[idx]
        waveform, _ = load_waveform(audio_path, sample_rate=SAMPLE_RATE)
        pitch[session] = {}
        for speaker in ["A", "B"]:
            channel = 0 if speaker == "A" else 1
            vad_list = torch.tensor(d["vad_list"][channel])

            tmp_pitch = []
            for start_time, end_time in vad_list:
                start = (start_time * SAMPLE_RATE).long()
                end = (end_time * SAMPLE_RATE).long()
                if end - start < min_samples:
                    continue
                y = waveform[channel, start:end]
                try:
                    p = pitch_praat(y, sample_rate=SAMPLE_RATE)
                except:
                    print("y: ", tuple(y.shape))
                    assert False
                tmp_pitch.append(p[p != 0])
            pp = torch.cat(tmp_pitch)
            pitch[session][speaker] = {
                "f0_mean": pp.mean().item(),
                "f0_std": pp.std().item(),
            }
        return pitch

    pitch = {}
    for session in tqdm(
        test_sessions, total=len(test_sessions), desc="extract speaker session prosody"
    ):
        if session not in pitch:
            pitch = extract_pitch_from_session(session, pitch)

    write_json(pitch, savepath)
    print("saved pitch information -> ", savepath)
    return pitch


def extract_fillers_pitch(savepath=None):
    fillers = read_txt(UH_PATH)
    filler_names = ["uh"] * len(fillers)
    um_fillers = read_txt(UM_PATH)
    filler_names += ["um"] * len(um_fillers)
    fillers += um_fillers
    pitch = read_json("data/pitch_information_test.json")

    min_samples = 800

    filler_f0 = {"uh": [], "um": []}
    for uh_or_um, filler in tqdm(
        zip(filler_names, fillers), total=len(fillers), desc="extract g"
    ):
        session, fill_start, fill_end, speaker = filler.split()
        start_time = float(fill_start)
        end_time = float(fill_end)
        if end_time - start_time < 0.01:
            continue
        channel = 0 if speaker == "A" else 1
        audio_path = join(AUDIO_ROOT, session_to_rel_path[session] + ".wav")
        y, _ = load_waveform(audio_path, start_time=start_time, end_time=end_time)
        if y.shape[-1] < min_samples:
            dy = min_samples - y.shape[-1]
            y = torch.cat((y, torch.zeros((2, dy))), dim=-1)
        p = pitch_praat(y[channel], sample_rate=SAMPLE_RATE)
        # Standardize the filler F0 by speaker
        # only voiced
        p = p[p != 0]
        m = pitch[session][speaker]["f0_mean"]
        s = pitch[session][speaker]["f0_std"]
        f0_norm = (p - m) / s
        filler_f0[uh_or_um].append(f0_norm.mean())

    f = {
        "uh": torch.stack(filler_f0["uh"]),
        "um": torch.stack(filler_f0["um"]),
    }
    if savepath is not None:
        torch.save(f, savepath)
    return f


if __name__ == "__main__":

    session_to_rel_path = read_json(REL_PATH)

    f = extract_fillers_pitch("data/average_standardized_pitch.pt")

    f = torch.load("data/average_standardized_pitch.pt")

    fig, ax = plt.subplots(2, 1, sharex=True)
    w_uh = torch.ones_like(f["uh"]) / f["uh"].shape[0]
    w_um = torch.ones_like(f["um"]) / f["um"].shape[0]
    ax[0].set_title("F0 (standardized) Fillers")
    ax[0].hist(
        f["uh"], bins=100, range=(-3, 3), weights=w_uh, label="UH", color="b", alpha=0.6
    )
    ax[1].hist(
        f["um"], bins=100, range=(-3, 3), weights=w_um, label="UM", color="g", alpha=0.6
    )
    ax[0].axvline(0, color="k", linestyle="dashed", linewidth=1)
    ax[1].axvline(0, color="k", linestyle="dashed", linewidth=1)
    ax[0].set_ylim([0, 0.09])
    ax[1].set_ylim([0, 0.09])
    ax[0].legend()
    ax[1].legend()
    fig.savefig("data/images/f0_standardize_filler.png")
    plt.show()
