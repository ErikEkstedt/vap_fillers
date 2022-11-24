import torch
import matplotlib.pyplot as plt


def plot_result(
    result,
    filler,
    frame_hz=50,
    col_fill="r",
    col_nofill="b",
    area_alpha=0.01,
    plot=True,
):
    pn_fill = result["filler"]["p_now"].mean(dim=0)[:-1]
    pn_fill_s = result["filler"]["p_now"].std(dim=0)[:-1]

    pf_fill = result["filler"]["p_fut"].mean(dim=0)[:-1]
    pf_fill_s = result["filler"]["p_fut"].std(dim=0)[:-1]

    ph_fill = result["filler"]["H"].mean(dim=0)[:-1]
    ph_fill_s = result["filler"]["H"].std(dim=0)[:-1]

    pn_nofill = result["no_filler"]["p_now"].mean(dim=0)[:-1]
    pn_nofill_s = result["no_filler"]["p_now"].std(dim=0)[:-1]

    pf_nofill = result["no_filler"]["p_fut"].mean(dim=0)[:-1]
    pf_nofill_s = result["no_filler"]["p_fut"].std(dim=0)[:-1]

    ph_nofill = result["no_filler"]["H"].mean(dim=0)[:-1]
    ph_nofill_s = result["no_filler"]["H"].std(dim=0)[:-1]

    x = torch.arange(len(pn_nofill)) / frame_hz

    fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # FILLER
    ax[0].set_title(f'"{filler}": HOLD prob')
    ax[0].plot(x, pn_fill, label="P now FILLER", color=col_fill)
    ax[0].fill_between(
        x, pn_fill - pn_fill_s, pn_fill + pn_fill_s, alpha=area_alpha, color=col_fill
    )
    # NO-FILLER
    ax[0].plot(x, pn_nofill, label="P now", color=col_nofill)
    ax[0].fill_between(
        x,
        pn_nofill - pn_nofill_s,
        pn_nofill + pn_nofill_s,
        alpha=area_alpha,
        color=col_nofill,
    )
    ax[0].set_ylim([-0.05, 1.05])

    ax[1].plot(x, pf_fill, label="P future FILLER", color=col_fill, linestyle="dashed")
    ax[1].fill_between(
        x, pf_fill - pf_fill_s, pf_fill + pf_fill_s, alpha=area_alpha, color=col_fill
    )
    ax[1].plot(x, pf_nofill, label="P future", color=col_nofill, linestyle="dashed")
    ax[1].fill_between(
        x,
        pf_nofill - pf_nofill_s,
        pf_nofill + pf_nofill_s,
        alpha=area_alpha,
        color=col_nofill,
    )
    ax[1].set_ylim([-0.05, 1.05])
    # DIFF
    ax[2].plot(x, pn_fill - pn_nofill, label="Diff now", color="k")
    ax[2].plot(
        x, pf_fill - pf_nofill, label="Diff future", color="k", linestyle="dashed"
    )
    ax[2].axhline(0, color="k", linewidth=1, linestyle="dashed")

    # Entropy, H
    ax[-1].plot(x, ph_fill - ph_nofill, label="H diff", color="k")
    ax[-1].plot(x, ph_fill, label="H FILLER", color=col_fill)
    ax[-1].plot(x, ph_nofill, label="H", color=col_nofill)
    # ax[-1].fill_between(
    #     x, ph_fill - ph_fill_s, ph_fill + ph_fill_s, alpha=area_alpha, color=col_fill
    # )
    # ax[-1].fill_between(
    #     x,
    #     ph_nofill - ph_nofill_s,
    #     ph_nofill + ph_nofill_s,
    #     alpha=area_alpha,
    #     color=col_nofill,
    # )
    ax[-1].axhline(0, color="k", linewidth=1, linestyle="dashed")
    ymin, _ = ax[-1].get_ylim()
    ax[-1].set_ylim([ymin, 8])
    for a in ax:
        a.legend(loc="upper right")
    # for a in ax[:-1]:
    #     # a.set_xticks([])
    #     a.set_ylim([-0.05, 1.1])
    plt.tight_layout()

    if plot:
        plt.pause(0.01)
    return fig, ax


def plot_diff(dm, fdm=None, filler="UM", plot=True):
    x = torch.arange(dm.shape[0]) / 50
    fig, ax = plt.subplots(1, 1)
    ax.plot(x[:-1], dm[:-1], label="diff: filler - nofiller")
    if fdm is not None:
        ax.plot(x[:-1], fdm[:-1], label="FUT diff: filler - nofiller")
    ax.axhline(0, color="k")
    ax.legend()
    ax.set_title(f'"{filler}": HOLD prob difference')
    ax.set_xlabel("seconds")
    ax.set_ylabel("HOLD %")
    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig, ax


if __name__ == "__main__":

    result = torch.load("filler_um_data.pt")
    # result = torch.load("filler_uh_data.pt")
    fig, ax = plot_result(result, filler="UM", frame_hz=50, area_alpha=0.05, plot=False)
    plt.show()

    diff = result["filler"]["p_now"] - result["no_filler"]["p_now"]
    fdiff = result["filler"]["p_fut"] - result["no_filler"]["p_fut"]
    dm = diff.mean(0) * 100
    fdm = fdiff.mean(0) * 100

    _ = plot_diff(dm, fdm, filler="UM", plot=False)
    plt.show()

    _ = plot_diff(fdm)
