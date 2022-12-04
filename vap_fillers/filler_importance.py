import torch
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    p = torch.load("data/filler_uh.pt")

    frame_hz = 50
    end_frames = [50, 100, 150, 200, 250, 300]

    pnf = p["filler"]["p_now"]
    pnn = p["no_filler"]["p_now"]

    for end_time in range(1, 6):
        end_frame = int(end_time * frame_hz)
        start_frame = end_frame - frame_hz
        print(start_frame, end_frame)
        pnf = p["filler"]["p_now"][:, start_frame:end_frame].mean(dim=-1)
        pnn = p["no_filler"]["p_now"][:, start_frame:end_frame].mean(dim=-1)
        shift_fill = (pnf < 0.5).sum()
        shift_nofill = (pnn < 0.5).sum()
        r = shift_nofill / shift_fill
        print(
            f"time {end_time}, filler: {shift_fill} No-filler: {shift_nofill}, r: {r}"
        )

    # Last frame contains waveform-cutoff noise
    end_time = 5.5
    end_frame = int(end_time * frame_hz)
    range = (-(end_time - 0.1), end_time - 0.1)
    pnf = p["filler"]["p_now"][:, :end_frame]
    pnf_to_first_shift = (pnf <= 0.5).cumsum(-1).clamp(max=1).sum(-1)
    pnn = p["no_filler"]["p_now"][:, :end_frame]
    pnn_to_first_shift = (pnn <= 0.5).cumsum(-1).clamp(max=1).sum(-1)
    diff = (pnn_to_first_shift - pnf_to_first_shift) / frame_hz  # to seconds
    #
    cutoff = 0.2
    interesting = diff[torch.where(diff.abs() >= cutoff)]
    same = diff[torch.where(diff.abs() < cutoff)]
    ni = interesting.nelement()
    ns = same.nelement()
    #
    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    nover, bins, _ = ax.hist(
        interesting[interesting > 0],
        bins=100,
        range=range,
        color="g",
        alpha=0.5,
        label="expected",
    )
    nunder, bins, _ = ax.hist(
        interesting[interesting < 0],
        bins=100,
        range=range,
        color="r",
        alpha=0.5,
        label="reverse",
    )
    # ax.hist(same, bins=100, range=(-4, 4), color='g', alpha=0.5)
    ax.set_xlabel("(nofiller-filler) shift time")
    ax.legend()
    ax.set_title(
        f"Max: {end_time}s | z-pad {cutoff}s ({n} vs {ni}) | nofill: {nover.sum()} | fill: {nunder.sum()}"
    )
    plt.show()
