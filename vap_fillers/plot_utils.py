import torch
import matplotlib.pyplot as plt

from vap.audio import log_mel_spectrogram


def plot_speaker_probs(x, p, ax, label="P", alpha=0.6, colors=["b", "orange"]):
    px = p - 0.5
    z = torch.zeros_like(p)
    ax.fill_between(
        x, px, z, where=px > 0, color=colors[0], label=f"A {label}", alpha=alpha
    )
    ax.fill_between(
        x, px, z, where=px < 0, color=colors[1], label=f"B {label}", alpha=alpha
    )
    ax.set_ylim([-0.5, 0.5])
    return ax


def plot_bc_probs(x, p_bc, ax, alpha=0.6):
    z = torch.zeros_like(p_bc[:, 0])
    ax.fill_between(x, p_bc[:, 0], z, color="b", label=f"A Short", alpha=alpha)
    ax.fill_between(x, -p_bc[:, 1], z, color="orange", label=f"B Short", alpha=alpha)
    ax.set_ylim([-1, 1])


def plot_entropy(x, h, ax):
    ax.plot(x, h.cpu(), label="H, entropy", color="green", linewidth=3)
    ax.set_ylim([0, 8])


def plot_waveform(y, sr=16_000, max_points=1000, ax=None):
    assert (
        y.ndim == 2
    ), f"Expects waveform of shape (N_CHANNELS, N_SAMPLES) but got {y.shape}"
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    else:
        fig = None

    n_channels = y.shape[0]

    hop_length = max(1, y.shape[-1] // max_points)
    y_env = y.unfold(dimension=-1, size=hop_length, step=hop_length)
    y_env = y_env.abs().max(dim=-1).values

    duration = y.shape[-1] / sr
    n_frames = y_env.shape[-1]
    s_per_frame = duration / n_frames
    x = torch.arange(0, duration, s_per_frame)
    x = x[:n_frames]
    ax.fill_between(x, -y_env[0], y_env[0], alpha=0.6, color="b", label="A")
    if n_channels > 1:
        ax.fill_between(x, -y_env[1], y_env[1], alpha=0.6, color="orange", label="B")
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlim([x[0], x[-1]])
    plt.tight_layout()
    return fig, ax


def plot_mel_spectrogram(
    y, sample_rate=16_000, hop_time=0.02, frame_time=0.05, n_mels=80, ax=None
):
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    else:
        fig = None

    duration = y.shape[-1] / sample_rate
    xmin, xmax = 0, duration
    ymin, ymax = 0, 80

    hop_length = round(sample_rate * hop_time)
    frame_length = round(sample_rate * frame_time)
    spec = log_mel_spectrogram(
        y,
        n_mels=n_mels,
        n_fft=frame_length,
        hop_length=hop_length,
        sample_rate=sample_rate,
    )
    ax[0].imshow(
        spec[0],
        interpolation="none",
        aspect="auto",
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
    )
    ax[1].imshow(
        spec[1],
        interpolation="none",
        aspect="auto",
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
    )

    if fig is not None:
        plt.subplots_adjust(
            left=0.05, bottom=None, right=0.99, top=0.99, wspace=0.01, hspace=0
        )
    return fig


def plot_words(tg_words, ax, fs=14):
    N = 8
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    ymax_quarter = span * 3 / 4
    yd = (ymax_quarter - ymin) / N

    # ymid = (ymax - ymin) / 2
    for ii, word in enumerate(tg_words):
        s, e, w = word
        s = float(s)
        e = float(e)
        mid = s + (e - s) / 2
        y = ymax_quarter - (ii % N) * yd
        ax.text(
            x=mid,
            y=y,
            s=w,
            color="w",
            fontsize=fs,
            fontweight="bold",
            rotation=45,
            horizontalalignment="center",
        )
        # ax.text(x=s, y=ymid, s=w, color='w', fontsize=fs, rotation=45, horizontalalignment='left')
        ax.axvline(s, linewidth=1, color="w")
        ax.axvline(e, linewidth=1, color="w")
