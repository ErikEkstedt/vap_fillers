import torch
import matplotlib.pyplot as plt
from os.path import join
import glob

from vap.audio import load_waveform

from vap_fillers.utils import read_text_grid, wav_path_to_tg_path
from vap_fillers.plot_utils import plot_mel_spectrogram, plot_waveform, plot_words


DIRPATH = "data/where_is_the_hesitation/Stimuli"
SAMPLE_RATE = 16_000

"""
The forced aligner does not seem to recognize the fille "eh" but do well on "uh" and 'um'
so lets try to change those...

Using 'uh' is much better but Montreal-Forced-Aligner still misses alot of 'initial_filler'
especially when followed by "I"...

However, the initial_filler is not that import for us... I think...
"""


if __name__ == "__main__":

    wavpaths = glob.glob(join(DIRPATH, "**/*.wav"), recursive=True)

    # single sample
    # wavpath = wavpaths[0]
    # audio_info = get_audio_info(wavpath)
    # tg_path = wav_path_to_tg_path(wavpath)
    # d = read_text_grid(tg_path)

    for wavpath in wavpaths:
        if "initial_filler" in wavpath:
            continue
        plt.close("all")
        waveform, _ = load_waveform(wavpath, sample_rate=SAMPLE_RATE)
        tg_path = wav_path_to_tg_path(wavpath)
        d = read_text_grid(tg_path)
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        plot_waveform(waveform, ax=ax[0])
        plot_mel_spectrogram(waveform, ax=[ax[1]])
        plot_words(d["words"], ax[1])
        plt.show()
