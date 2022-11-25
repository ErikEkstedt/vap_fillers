import torch
from os.path import join, basename, dirname
from os import makedirs
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

from vap.audio import load_waveform
from vap_fillers.utils import (
    load_model,
    read_text_grid,
    wav_path_to_tg_path,
    pad_silence,
)
from vap.utils import write_json
from vap_fillers.plot_utils import plot_mel_spectrogram, plot_words

DIRPATH = "data/where_is_the_hesitation/Stimuli"
MODELOUT = "data/where_is_the_hesitation/Model"

"""

wavpath: 
misunderstandings_0_-1,5_initial.wav' = name_{speech-rate}_{pitch}_{filler_placement}

"""


def get_sample_info(wavpath):
    name, rate_str, pitch_str, filler_pos = (
        basename(wavpath).replace(".wav", "").split("_")
    )
    rate = "slow"
    if rate_str == "0":
        rate = "medium"
    elif rate_str == "1,5":
        rate = "fast"
    pitch = "low"
    if pitch_str == "0":
        pitch = "medium"
    elif pitch_str == "1,5":
        pitch = "high"
    return name, rate, pitch, filler_pos


def condense_data(d):
    new_d = {}
    for key, values in d.items():
        new_d[key] = torch.stack(values).mean()
    return new_d


def plot_model_sample(d, out):
    fig, ax = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    # plot_waveform(waveform, ax=ax[0])
    plot_mel_spectrogram(waveform[:, 0], ax=[ax[0]])
    plot_words(d["words"], ax[0])
    ax[0].set_ylabel("Mels")
    # plot model output
    x = torch.arange(out["p_now"].shape[1]) / model.frame_hz
    ax[1].plot(
        x,
        out["p_now"][0, :, 0].cpu(),
        color="r",
        linewidth=2,
        label="HOLD now",
        alpha=0.6,
    )
    ax[1].plot(
        x,
        out["p_future"][0, :, 0].cpu(),
        color="r",
        linestyle="dashed",
        linewidth=2,
        label="HOLD now",
        alpha=0.6,
    )
    ax[1].axhline(0.5, linewidth=1, linestyle="dashed", color="k")
    ax[1].set_ylabel("HOLD prob")
    ax[1].legend(loc="lower left")
    a = ax[1].twinx()
    a.plot(x, out["H"][0].cpu(), color="g", linewidth=2, label="Entropy")
    a.set_ylim([0, 8])
    a.legend(loc="lower right")
    a.set_ylabel("Entropy")
    ax[1].set_ylim([0, 1])
    ax[1].set_xlabel("seconds")
    plt.subplots_adjust(left=0.05, bottom=None, right=0.95, top=0.99, hspace=0.03)

    return fig


if __name__ == "__main__":

    model = load_model()
    wavpaths = glob.glob(join(DIRPATH, "**/*.wav"), recursive=True)

    makedirs(MODELOUT, exist_ok=True)
    all_out = {"wavpaths": [], "p_now": [], "p_future": [], "H": []}
    pad_frame = 0
    entropy_filler = {"none": [], "medial": [], "initial": []}
    entropy_rate = {"slow": [], "medium": [], "fast": []}
    entropy_pitch = {"low": [], "medium": [], "high": []}
    for wavpath in tqdm(wavpaths):
        name, rate, pitch, filler_pos = get_sample_info(wavpath)
        waveform, _ = load_waveform(wavpath, sample_rate=model.sample_rate)
        tg_path = wav_path_to_tg_path(wavpath)
        d = read_text_grid(tg_path)
        # Pad as stereo channels
        waveform = model._pad_zero_channel(waveform.unsqueeze(1))
        # add zeros
        waveform = pad_silence(waveform, silence=3)
        # Model forward
        out = model.probs(waveform.to(model.device))
        filename = basename(wavpath).replace(".wav", ".json")
        folder = basename(dirname(wavpath))
        dirpath = join(MODELOUT, folder)
        makedirs(dirpath, exist_ok=True)
        tmp_data = {
            "wavpath": wavpath,
            "p_now": out["p_now"][0, :, 0].cpu().tolist(),
            "p_future": out["p_future"][0, :, 0].cpu().tolist(),
            "H": out["H"][0].cpu().tolist(),
        }
        write_json(tmp_data, join(dirpath, filename))
        ###############################################
        # metrics
        ###############################################
        start = d["words"][0][0]
        end = d["words"][-1][1]
        start_frame = round(start * model.frame_hz)
        end_frame = round(end * model.frame_hz)
        ent = out["H"][0, start_frame + pad_frame : end_frame - pad_frame].cpu().mean()
        entropy_filler[filler_pos].append(ent)
        entropy_rate[rate].append(ent)
        entropy_pitch[pitch].append(ent)

        ###############################################
        # Figure
        ###############################################
        plt.close("all")
        fig = plot_model_sample(d, out)
        plt.show()

    avg_entropy = {
        "filler": condense_data(entropy_filler),
        "rate": condense_data(entropy_rate),
        "pitch": condense_data(entropy_pitch),
    }

    # ENTROPY w.r.t turn-taking dependent on filler placement
    # average over average entropy:
    # PAD = 0
    # {'filler': {'none': tensor(2.1311),
    #   'medial': tensor(2.4649),
    #   'initial': tensor(2.6237)},
    #  'rate': {'slow': tensor(2.4204),
    #   'medium': tensor(2.3603),
    #   'fast': tensor(2.4398)},
    #  'pitch': {'low': tensor(2.5447),
    #   'medium': tensor(2.3666),
    #   'high': tensor(2.2956)}}
    ########################################
    # PAD = 5
    # {'filler': {'none': tensor(2.0654),
    #   'medial': tensor(2.4401),
    #   'initial': tensor(2.5577)},
    #  'rate': {'slow': tensor(2.3656),
    #   'medium': tensor(2.3104),
    #   'fast': tensor(2.3882)},
    #  'pitch': {'low': tensor(2.4893),
    #   'medium': tensor(2.3135),
    #   'high': tensor(2.2480)}}
