from argparse import ArgumentParser
from os.path import expanduser, join
from tqdm import tqdm
import pandas as pd
import torch

from vap.audio import load_waveform
from vap.utils import read_json
from vap_fillers.utils import load_model, pad_silence


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-fp",
        "--filler_path",
        type=str,
        default="results/all_fillers_test_prosody.csv",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="../VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=join(expanduser("~"), "projects/data/switchboard/audio"),
        help="Path to swb audio",
    )
    parser.add_argument(
        "--rel_path",
        type=str,
        default="data/relative_audio_path.json",
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
    args = parser.parse_args()
    print("-" * 50)
    print("ARGUMENTS")
    print("---------")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-" * 50)
    return args


def find_cross(p, cutoff=0.5):
    assert p.ndim == 1, f"Expects (n_frames,) got {p.shape}"
    cross_idx = torch.where(p <= cutoff)[0]
    cross = -1
    if len(cross_idx) > 0:
        cross = cross_idx[0].cpu().item()
    return cross


if __name__ == "__main__":

    args = get_args()

    model = load_model(args.checkpoint)
    filler_df = pd.read_csv(args.filler_path)
    rel_paths = read_json(args.rel_path)

    new_df = []
    for row_idx in tqdm(range(len(filler_df)), desc="Extract crosses"):
        filler = filler_df.loc[row_idx]
        if filler.start < args.context:
            continue

        # Frames and samples
        start = filler.start - args.context
        filler_dur = filler.end - filler.start
        filler_samples = round(filler_dur * model.sample_rate)
        filler_frames = round(filler_dur * model.frame_hz)
        sil_start_omit = round(args.context * model.frame_hz)
        sil_start_filler = sil_start_omit + filler_frames

        # Load audio
        audio_path = join(args.audio_path, rel_paths[str(filler.session)] + ".wav")
        waveform, _ = load_waveform(
            audio_path,
            start_time=start,
            end_time=filler.end,
            sample_rate=model.sample_rate,
        )
        waveform = waveform.unsqueeze(0)
        wav_filler = pad_silence(
            waveform, silence=args.silence, sample_rate=model.sample_rate
        )

        ##################################################
        # Combine filler/omit filler
        ##################################################
        wav_omit = waveform[..., :-filler_samples]
        diff = wav_filler.shape[-1] - wav_omit.shape[-1]
        wav_omit = pad_silence(wav_omit, sil_samples=diff)

        ##################################################
        # Combine filler/omit filler
        ##################################################
        wav_replace = waveform[..., :-filler_samples]
        replacement = waveform[..., -int(2 * filler_samples) : -filler_samples]
        wav_replace = torch.cat((wav_replace, replacement), dim=-1)
        diff = wav_replace.shape[-1] - wav_omit.shape[-1]
        wav_omit = pad_silence(wav_omit, sil_samples=diff)

        ##################################################
        # Forward
        ##################################################
        y = torch.cat([wav_filler, wav_omit])
        out = model.probs(y.to(model.device))

        # Extract crosses
        speaker_idx = 0 if filler.speaker == "A" else 1
        pnf = out["p_now"][0, sil_start_filler:, speaker_idx]
        pff = out["p_now"][0, sil_start_filler:, speaker_idx]
        pno = out["p_now"][1, sil_start_omit:-filler_frames, speaker_idx]
        pfo = out["p_now"][1, sil_start_omit:-filler_frames, speaker_idx]

        new_filler = filler.to_dict()
        new_filler["filler_now_cross"] = find_cross(pnf)
        new_filler["filler_fut_cross"] = find_cross(pff)
        new_filler["omit_now_cross"] = find_cross(pno)
        new_filler["omit_fut_cross"] = find_cross(pfo)
        new_df.append(new_filler)

    df = pd.DataFrame(new_df)
    savepath = args.filler_path.replace(".csv", "_model.csv")
    df.to_csv(savepath)
    print("Saved -> ", savepath)
