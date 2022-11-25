import torch
from os.path import join
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
from vap_fillers.plot_utils import plot_waveform, plot_mel_spectrogram, plot_words

DIRPATH = "data/where_is_the_hesitation/Stimuli"

if __name__ == "__main__":

    model = load_model()
    wavpaths = glob.glob(join(DIRPATH, "**/*.wav"), recursive=True)

    pad_frame = 0

    entropy = {"no_filler": [], "initial_filler": [], "medial_filler": []}
    for wavpath in tqdm(wavpaths):
        # if "initial_filler" in wavpath:
        #     continue
        waveform, _ = load_waveform(wavpath, sample_rate=model.sample_rate)
        tg_path = wav_path_to_tg_path(wavpath)
        d = read_text_grid(tg_path)
        # Pad as stereo channels
        waveform = model._pad_zero_channel(waveform.unsqueeze(1))
        # add zeros
        waveform = pad_silence(waveform, silence=3)
        # Model forward
        out = model.probs(waveform.to(model.device))
        ###############################################
        # metrics
        ###############################################
        start = d["words"][0][0]
        end = d["words"][-1][1]
        start_frame = round(start * model.frame_hz)
        end_frame = round(end * model.frame_hz)
        ent = out["H"][0, start_frame + pad_frame : end_frame - pad_frame].cpu().mean()
        if "no_filler" in wavpath:
            entropy["no_filler"].append(ent)
        elif "initial_filler" in wavpath:
            entropy["initial_filler"].append(ent)
        else:
            entropy["medial_filler"].append(ent)

        ###############################################
        # Figure
        ###############################################
        plt.close("all")
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
        plt.show()

    avg_ent = {}
    for filler, data in entropy.items():
        print(filler, len(data))
        avg_ent[filler] = torch.stack(data).mean()

    # ENTROPY w.r.t turn-taking dependent on filler placement
    # average over average entropy:
    # {'no_filler': tensor(2.1311),
    #  'initial_filler': tensor(2.6237),
    #  'medial_filler': tensor(2.4649)}
