from argparse import ArgumentParser
from os.path import expanduser, join
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from vap.audio import load_waveform
from vap.model import VAPModel
from vap.utils import everything_deterministic, read_txt, read_json

everything_deterministic()
torch.manual_seed(0)

REL_PATH = "data/relative_audio_path.json"
TEST_FILE_PATH = "data/test.txt"


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="../VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt",
        help="Path to trained model",
    )
    parser.add_argument(
        "-f",
        "--filler_path",
        type=str,
        default="data/uh.txt",
        help="Path to txt with fillers",
    )
    parser.add_argument(
        "-r",
        "--audio_root",
        type=str,
        default=join(expanduser("~"), "projects/data/switchboard/audio"),
        help="Path to swb audio",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="filler_data.json",
        help="filename to save data to",
    )
    parser.add_argument(
        "--fig_path",
        type=str,
        default=None,
        help="filename to save figure",
    )
    parser.add_argument(
        "--context",
        type=float,
        default=20,
        help="Duration of each chunk processed by model",
    )
    parser.add_argument(
        "--silence",
        type=float,
        default=10,
        help="Duration of silence after the fillers/no-filler",
    )
    args = parser.parse_args()
    return args


def load_model(checkpoint):
    print("Load Model...")
    model = VAPModel.load_from_checkpoint(checkpoint)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def pad_silence(waveform, silence=10, sample_rate=16_000):
    assert (
        waveform.ndim == 3
    ), f"Expects waveform of shape (B, C, n_samples) but got {waveform.shape}"
    B, C, _ = waveform.size()
    sil_samples = int(silence * sample_rate)
    z = torch.zeros((B, C, sil_samples), device=waveform.device)
    return torch.cat([waveform, z], dim=-1)


def plot_result(result, col_fill="r", col_nofill="b", area_alpha=0.01, plot=True):
    pn_fill = result["filler"]["p_now"].mean(dim=0)
    pn_fill_s = result["filler"]["p_now"].std(dim=0)

    pf_fill = result["filler"]["p_fut"].mean(dim=0)
    pf_fill_s = result["filler"]["p_fut"].std(dim=0)

    ph_fill = result["filler"]["H"].mean(dim=0)
    ph_fill_s = result["filler"]["H"].std(dim=0)

    pn_nofill = result["no_filler"]["p_now"].mean(dim=0)
    pn_nofill_s = result["no_filler"]["p_now"].std(dim=0)

    pf_nofill = result["no_filler"]["p_fut"].mean(dim=0)
    pf_nofill_s = result["no_filler"]["p_fut"].std(dim=0)

    ph_nofill = result["no_filler"]["H"].mean(dim=0)
    ph_nofill_s = result["no_filler"]["H"].std(dim=0)

    x = torch.arange(len(pn_nofill)) / model.frame_hz

    area_alpha = 0.05
    fig, ax = plt.subplots(6, 1, figsize=(12, 12), sharex=True)

    # FILLER
    ax[0].plot(x, pn_fill, label="P now FILLER", color=col_fill)
    ax[0].fill_between(
        x, pn_fill - pn_fill_s, pn_fill + pn_fill_s, alpha=area_alpha, color=col_fill
    )
    ax[1].plot(x, pf_fill, label="P future FILLER", color=col_fill, linestyle="dashed")
    ax[1].fill_between(
        x, pf_fill - pf_fill_s, pf_fill + pf_fill_s, alpha=area_alpha, color=col_fill
    )

    # NO-FILLER
    ax[2].plot(x, pn_nofill, label="P now", color=col_nofill)
    ax[2].fill_between(
        x,
        pn_nofill - pn_nofill_s,
        pn_nofill + pn_nofill_s,
        alpha=area_alpha,
        color=col_nofill,
    )
    ax[3].plot(x, pf_nofill, label="P future", color=col_nofill, linestyle="dashed")
    ax[3].fill_between(
        x,
        pf_nofill - pf_nofill_s,
        pf_nofill + pf_nofill_s,
        alpha=area_alpha,
        color=col_nofill,
    )
    # DIFF
    ax[4].plot(x, pn_fill - pn_nofill, label="Diff now", color="k")
    ax[4].plot(
        x, pf_fill - pf_nofill, label="Diff future", color="k", linestyle="dashed"
    )

    # Entropy, H
    ax[-1].plot(x, ph_fill - ph_nofill, label="H diff", color="k")
    ax[-1].fill_between(
        x, ph_fill - ph_fill_s, ph_fill + ph_fill_s, alpha=area_alpha, color=col_fill
    )
    ax[-1].fill_between(
        x,
        ph_nofill - ph_nofill_s,
        ph_nofill + ph_nofill_s,
        alpha=area_alpha,
        color=col_nofill,
    )
    ax[-1].plot(x, ph_fill, label="H FILLER", color=col_fill)
    ax[-1].plot(x, ph_nofill, label="H", color=col_nofill)
    ax[-1].set_ylim([0, 8])
    for a in ax:
        a.legend(loc="upper right")
    for a in ax[:-1]:
        # a.set_xticks([])
        a.set_ylim([-0.05, 1.1])
    plt.tight_layout()

    if plot:
        plt.pause(0.01)
    return fig, ax


if __name__ == "__main__":
    args = get_args()
    session_to_rel_path = read_json(REL_PATH)
    model = load_model(args.checkpoint)
    fillers = read_txt(args.filepath)
    ok_files = read_txt(TEST_FILE_PATH)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("Fillers: ", len(fillers))
    sil_frames = int(args.silence * model.frame_hz)

    skipped_by_file = 0
    skipped_by_context = 0
    # TODO: should we save session and timestamps with the data?
    result = {
        "filler": {"type": [], "p_now": [], "p_fut": [], "H": []},
        "no_filler": {"type": [], "p_now": [], "p_fut": [], "H": []},
    }
    for filler in tqdm(fillers, desc="Extract filler probs"):
        session, fill_start, fill_end, speaker = filler.split()
        # session, fill_start, fill_end, speaker, fill_type = "2001", 44, 46, 0, "eh"

        # 3717 36.55 37.001 B

        if session not in ok_files:
            skipped_by_file += 1
            continue

        speaker = 0 if speaker == "A" else 1
        fill_start = float(fill_start)
        fill_end = float(fill_end)
        start = fill_start - args.context

        # We omit every filler that does not have the correct context
        if start < 0:
            skipped_by_context += 1
            continue

        ############################################################
        # Load waveform, add batch dim and move to correct device
        # add batch dimension -> (B, 2, n_samples)
        ############################################################
        audio_path = join(args.audio_root, session_to_rel_path[session] + ".wav")
        waveform, _ = load_waveform(audio_path, start_time=start, end_time=fill_end)
        waveform = waveform.unsqueeze(0).to(model.device)

        ############################################################
        # Include filler
        ############################################################
        wav_filler = pad_silence(
            waveform, silence=args.silence, sample_rate=model.sample_rate
        )

        out = model.probs(wav_filler)

        # result["filler"]["type"].append(fill_type)
        result["filler"]["p_now"].append(out["p_now"][0, -sil_frames:, speaker].cpu())
        result["filler"]["p_fut"].append(
            out["p_future"][0, -sil_frames:, speaker].cpu()
        )
        result["filler"]["H"].append(out["H"][0, -sil_frames:].cpu())

        ############################################################
        # Omit the filler
        ############################################################
        fill_dur = fill_end - fill_start
        fill_n_samples = int(fill_dur * model.sample_rate)
        w_no_filler = waveform[..., :-fill_n_samples]
        w_no_filler = pad_silence(
            w_no_filler, silence=args.silence, sample_rate=model.sample_rate
        )
        out = model.probs(w_no_filler)
        # result["no_filler"]["type"].append(fill_type)
        result["no_filler"]["p_now"].append(
            out["p_now"][0, -sil_frames:, speaker].cpu()
        )
        result["no_filler"]["p_fut"].append(
            out["p_future"][0, -sil_frames:, speaker].cpu()
        )
        result["no_filler"]["H"].append(out["H"][0, -sil_frames:].cpu())

    print("SKIPPED BY FILE: ", skipped_by_file)
    print("SKIPPED BY CONTEXT: ", skipped_by_context)

    ############################################################
    # Stack tensors
    ############################################################
    for fill_no_fill, data in result.items():
        for name, tensor_list in data.items():
            if name == "type":
                continue
            result[fill_no_fill][name] = torch.stack(tensor_list)

    torch.save(result, args.data_path)
    print("Saved all data -> ", args.output_path)

    ############################################################
    # Figure
    ############################################################
    if args.figpath is not None:
        fig, ax = plot_result(result)
        fig.savefig(args.fig_path)
