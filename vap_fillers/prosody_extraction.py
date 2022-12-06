import torch
import numpy as np
from tqdm import tqdm
from os.path import expanduser, join
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from datasets_turntaking.dialog_audio_dataset import load_spoken_dialog_audio_dataset
from vap.audio import load_waveform, get_audio_info
from vap.functional import pitch_praat, intensity_praat
from vap.utils import read_txt, read_json, write_txt, write_json


UH_ALL_PATH = "data/uh.txt"
UM_ALL_PATH = "data/um.txt"
UH_PATH = "data/uh_test.txt"
UM_PATH = "data/um_test.txt"
REL_PATH = "data/relative_audio_path.json"
TEST_FILE_PATH = "data/test.txt"
AUDIO_ROOT = join(expanduser("~"), "projects/data/switchboard/audio")
SAMPLE_RATE = 16_000

"""

"""


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


def _extract_global_speaker_session_pitch(savepath="data/pitch_information_test.json"):
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


def prosody_extraction():
    min_prosody_time = 0.11  # seconds to cleanly extract pitch/intensity

    min_p_samples = round(SAMPLE_RATE * min_prosody_time)

    # Helper to find audio files
    session_to_rel_path = read_json(REL_PATH)

    uh_fillers = read_txt(UH_PATH)
    um_fillers = read_txt(UM_PATH)
    fillers = uh_fillers + um_fillers

    # keep track of type of filler
    filler_names = ["uh"] * len(uh_fillers)
    filler_names += ["um"] * len(um_fillers)

    # Global pitch of each speaker in a session
    pitch = read_json("data/pitch_information_test.json")

    # Global intensity of each speaker in each session
    intensity = read_json("data/intensity_information_test.json")

    n_parts = 2
    skipped_pitch = 0
    skipped_ints = 0
    skipped_parts = 0
    skipped_nan = 0
    prosody = {
        "uh": {"pitch": [], "intensity": [], "duration": []},
        "um": {"pitch": [], "intensity": [], "duration": []},
    }
    for uh_or_um, filler in tqdm(zip(filler_names, fillers), total=len(fillers)):
        session, fill_start, fill_end, speaker = filler.split()
        start_time = float(fill_start)
        end_time = float(fill_end)
        duration = end_time - start_time
        if duration < 0.04:
            continue
        # load waveform and extract prosody
        audio_path = join(AUDIO_ROOT, session_to_rel_path[session] + ".wav")
        y, _ = load_waveform(
            audio_path,
            start_time=start_time,
            end_time=end_time,
            sample_rate=SAMPLE_RATE,
        )
        # Should we simply omit fillers shorter than
        # if y.shape[-1] < min_p_samples:
        #     print(f"too short {y.shape}")
        #     continue
        # diff = min_p_samples - y.shape[-1]
        # y = torch.cat((y, torch.ones((2, diff), dtype=y.dtype)), dim=-1)

        ###########################################
        # PITCH
        pm = pitch[session][speaker]["mean"]
        ps = pitch[session][speaker]["std"]
        try:
            p = pitch_praat(y, hop_time=0.01, sample_rate=SAMPLE_RATE)
        except:
            skipped_pitch += 1
            continue
        p = p[p != 0]
        p = (p - pm) / ps
        if len(p) == 0:
            print("skipped len")
            continue
        if p.isnan().sum() > 0:
            print("pitch NaN")
            skipped_nan += 1
            continue
        # Divide voiced frames in 3 parts in time
        # get average pitch and store as a tensor
        # something like _-_ or __- or --_ etc
        # some dynamical envelope approximation
        pp = torch.stack([p_tmp.mean() for p_tmp in p.chunk(n_parts)])

        if pp.shape[-1] != n_parts:
            skipped_parts += 1
            # print("pitch chunks: ", pp.shape)
            continue

        ###########################################
        # INTENTISY
        inm = intensity[session][speaker]["mean"]
        ins = intensity[session][speaker]["std"]
        try:
            i = intensity_praat(y, hop_time=0.01, sample_rate=SAMPLE_RATE)
        except:
            skipped_ints += 1
            continue
        i = i[i != 0]
        i = (i - pm) / ps
        if i.isnan().sum() > 0:
            print("intensity NaN")
            skipped_nan += 1
            continue
        ii = torch.stack([i_tmp.mean() for i_tmp in i.chunk(n_parts)])
        if ii.shape[-1] != n_parts:
            skipped_parts += 1
            # print("intens chunks: ", ii.shape)
            # print("-" * 20)
            continue

        ###########################################
        prosody[uh_or_um]["pitch"].append(pp)
        prosody[uh_or_um]["intensity"].append(ii)
        prosody[uh_or_um]["duration"].append(duration)

    pp = {
        "uh": {
            "pitch": torch.stack(prosody["uh"]["pitch"]),
            "intensity": torch.stack(prosody["uh"]["intensity"]),
            "duration": torch.tensor(prosody["uh"]["duration"]).unsqueeze(-1),
        },
        "um": {
            "pitch": torch.stack(prosody["um"]["pitch"]),
            "intensity": torch.stack(prosody["um"]["intensity"]),
            "duration": torch.tensor(prosody["um"]["duration"]).unsqueeze(-1),
        },
    }

    # pp['um']['pitch']:        (N, n_parts)
    # pp['um']['intensity']:    (N, n_parts)
    # pp['um']['duration']:     (N, 1)

    # filler_pros: 7,

    torch.save(pp, "data/prosody.pt")

    return pp


def dim_reduction(pros):

    pros = torch.load("data/prosody.pt")

    from sklearn.manifold import TSNE

    uh_pros = torch.cat(
        (pros["uh"]["pitch"], pros["uh"]["intensity"], pros["uh"]["duration"]), dim=-1
    )
    um_pros = torch.cat(
        (pros["um"]["pitch"], pros["um"]["intensity"], pros["um"]["duration"]), dim=-1
    )

    fig, ax = plt.subplots(1, 1)
    ax.set_title("DURATION")
    ax.hist(pros["uh"]["duration"][:, 0], color="g", bins=100, alpha=0.2, label="uh")
    ax.hist(pros["um"]["duration"][:, 0], color="r", bins=100, alpha=0.2, label="um")
    ax.legend()
    ax.set_xlabel("duration")
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.set_title("dPITCH")
    ax.hist(
        pros["uh"]["pitch"][:, 1] - pros["uh"]["pitch"][:, 0],
        color="g",
        bins=100,
        alpha=0.2,
        label="uh",
        range=(-2.5, 2.5),
    )
    ax.hist(
        pros["um"]["pitch"][:, 1] - pros["um"]["pitch"][:, 0],
        color="r",
        bins=100,
        alpha=0.2,
        label="um",
        range=(-2.5, 2.5),
    )
    ax.legend()
    ax.set_xlabel("pitch diff (end - start)")
    plt.show()

    # PITCH start/end
    fig, ax = plt.subplots(1, 1)
    ax.scatter(
        pros["uh"]["pitch"][:, 0],
        pros["uh"]["pitch"][:, 1],
        color="g",
        label="uh",
        alpha=0.15,
    )
    ax.scatter(
        pros["um"]["pitch"][:, 0],
        pros["um"]["pitch"][:, 1],
        color="r",
        label="um",
        alpha=0.15,
    )
    # xmin, xmax = ax.get_xlim()
    # ymin, ymax = ax.get_ylim()
    ax.plot([-5, 12.5], [-5, 12.5], color="k", label="flat")
    ax.legend()
    ax.set_title("PITCH")
    ax.set_xlabel("start")
    ax.set_ylabel("end")
    plt.show()

    # INTENSITY start/end
    fig, ax = plt.subplots(1, 1)
    ax.scatter(
        pros["uh"]["intensity"][:, 0],
        pros["uh"]["intensity"][:, 1],
        color="g",
        label="uh",
        alpha=0.15,
    )
    ax.scatter(
        pros["um"]["intensity"][:, 0],
        pros["um"]["intensity"][:, 1],
        color="r",
        label="um",
        alpha=0.15,
    )
    ax.plot([-20, 0], [-20, 0], color="k", label="flat")
    ax.set_xlim([-20, 0])
    ax.set_ylim([-20, 0])
    ax.legend()
    ax.set_title("INTENSITY")
    ax.set_xlabel("start")
    ax.set_ylabel("end")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pros["uh"]["intensity"].mean(-1),
        pros["uh"]["pitch"].mean(-1),
        pros["uh"]["duration"],
        color="g",
        label="uh",
        alpha=0.15,
    )
    ax.scatter(
        pros["um"]["intensity"].mean(-1),
        pros["um"]["pitch"].mean(-1),
        pros["um"]["duration"],
        color="r",
        label="um",
        alpha=0.15,
    )
    ax.set_xlabel("intensity")
    ax.set_ylabel("pitch")
    ax.set_zlabel("duration")
    plt.show()

    fig, ax = plt.subplots(1, 1)
    # ax.scatter(pros['uh']['intensity'].mean(-1), pros['uh']['pitch'].mean(-1), color='g', label='uh', alpha=0.15)
    # ax.scatter(pros['um']['intensity'].mean(-1), pros['um']['pitch'].mean(-1), color='r', label='um', alpha=0.15)
    # ax.set_xlabel('intensity')
    ax.scatter(
        pros["uh"]["duration"],
        pros["uh"]["pitch"].mean(-1),
        color="g",
        label="uh",
        alpha=0.15,
    )
    ax.scatter(
        pros["um"]["duration"],
        pros["um"]["pitch"].mean(-1),
        color="r",
        label="um",
        alpha=0.15,
    )
    ax.set_xlabel("duration")
    ax.axvline(0, linestyle="dashed", color="k", linewidth=1)
    ax.axhline(0, linestyle="dashed", color="k", linewidth=1)
    ax.set_ylabel("pitch")
    ax.legend()
    plt.tight_layout()
    plt.show()

    uhs = uh_pros.shape[0]
    # ums = um_pros.shape[0]

    x = torch.cat((uh_pros, um_pros), dim=0)
    tsne = TSNE(n_components=2, perplexity=50, n_iter=3000, learning_rate="auto")
    xy = tsne.fit_transform(x)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(xy[:uhs, 0], xy[:uhs, 1], color="g", label="uh", alpha=0.3, s=5)
    # ax.scatter(xy[:uhs, 0], xy[:uhs, 1], color="g", label="uh", alpha=0.02, s=75)
    ax.scatter(xy[uhs:, 0], xy[uhs:, 1], color="r", label="um", alpha=0.3, s=5)
    # ax.scatter(xy[uhs:, 0], xy[uhs:, 1], color="r", label="um", alpha=0.02, s=75)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    session_to_rel_path = read_json(REL_PATH)

    # f = extract_fillers_pitch("data/average_standardized_pitch.pt")

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
