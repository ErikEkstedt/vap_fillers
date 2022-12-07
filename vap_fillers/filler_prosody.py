from argparse import ArgumentParser
from glob import glob
from os.path import basename, dirname, exists, join
from os import makedirs
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import parselmouth

from datasets_turntaking.dialog_audio_dataset import (
    load_spoken_dialog_audio_dataset,
)
from vap.audio import load_waveform
from vap.utils import read_json


SAMPLE_RATE = 16_000

ANNO_PATH = "../../data/switchboard/annotations/swb_ms98_transcriptions"
AUDIO_ROOT = "../../data/switchboard/audio"
REL_PATH = "data/relative_audio_path.json"


class Prosody:
    def __init__(
        self,
        audio_root: str = AUDIO_ROOT,
        sample_rate: int = SAMPLE_RATE,
        min_segment_time: float = 0.2,
    ):
        self.session_to_rel_path = read_json(REL_PATH)
        self.audio_root = audio_root
        self.sample_rate = sample_rate
        self.hop_time: float = 0.01
        self.f0_min: int = 60
        self.f0_max: int = 400
        self.min_samples = min_segment_time * sample_rate

    def torch_to_praat_sound(
        self, x: torch.Tensor, sample_rate: int = SAMPLE_RATE
    ) -> parselmouth.Sound:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy().astype("float64")
        return parselmouth.Sound(x, sampling_frequency=sample_rate)

    def praat_to_torch(self, sound: parselmouth.Sound) -> torch.Tensor:
        y = sound.as_array().astype("float32")
        return torch.from_numpy(y)

    def extract_prosody(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16_000,
        hop_time: float = 0.01,
        f0_min: int = 60,
        f0_max: int = 400,
    ):
        sound = self.torch_to_praat_sound(waveform, sample_rate)

        # F0
        pitch = sound.to_pitch(
            time_step=hop_time, pitch_floor=f0_min, pitch_ceiling=f0_max
        )
        intensity = sound.to_intensity(
            time_step=hop_time, minimum_pitch=f0_min, subtract_mean=True  # by default
        )

        # To tensor
        pitch = torch.from_numpy(pitch.selected_array["frequency"]).float()
        intensity = self.praat_to_torch(intensity)[0]
        return pitch, intensity

    def session_to_audio_path(self, session):
        return join(self.audio_root, self.session_to_rel_path[str(session)] + ".wav")

    def extract_prosody_from_session(self, d, session):
        """
        Extract global (mean, std) from all spoken segments, of each speaker, in the session.

        Returns:
            dict:
                session,
                a_f0_mean,
                a_f0_std,
                a_intensity_mean,
                a_intensity_std,
                b_f0_mean,
                b_f0_std,
                b_intensity_mean,
                b_intensity_std,
        """

        # Load Audio
        audio_path = self.session_to_audio_path(session)
        waveform, _ = load_waveform(audio_path, sample_rate=self.sample_rate)

        # Iterate over the voice activity from both speakers and extract prosodic features
        # Keep all relevant values (i.e. voiced frames for F0) and do statistics over all
        # speaker frames in the session
        session_prosody = {"session": session}
        for speaker in ["A", "B"]:
            channel = 0 if speaker == "A" else 1
            vad_list = torch.tensor(d["vad_list"][channel])

            speaker_pitch, speaker_intensity = [], []
            for start_time, end_time in vad_list:
                start = (start_time * self.sample_rate).round().long()
                end = (end_time * self.sample_rate).round().long()

                # Only use segments longer than `min_segment_time`
                # for extracting global prosody
                if end - start < self.min_samples:
                    continue

                # Extract prosody
                y = waveform[channel, start:end]
                p, ints = self.extract_prosody(
                    y,
                    sample_rate=self.sample_rate,
                    hop_time=self.hop_time,
                    f0_min=self.f0_min,
                    f0_max=self.f0_max,
                )

                # Add valid values
                speaker_pitch.append(p[p != 0])
                speaker_intensity.append(ints)

            # Average/standard deviation over all collected values
            speaker_pitch = torch.cat(speaker_pitch)
            speaker_intensity = torch.cat(speaker_intensity)
            session_prosody[f"{speaker.lower()}_f0_mean"] = speaker_pitch.mean().item()
            session_prosody[f"{speaker.lower()}_f0_std"] = speaker_pitch.std().item()
            session_prosody[
                f"{speaker.lower()}_intensity_mean"
            ] = speaker_intensity.mean().item()
            session_prosody[
                f"{speaker.lower()}_intensity_std"
            ] = speaker_intensity.std().item()

        return session_prosody

    def extract_swb_global_prosody(self, savepath):
        """
        Iterate over all files in the switchboard corpus and extract prosody
        (f0, intensity) statistics for each speaker in each session.
        """
        if not exists(ANNO_PATH):
            print("Annotations not found!")
            print(
                "Make sure you have `swb_ms98_transcriptions` downloaded and extracted"
            )
            print(ANNO_PATH)
            return

        all_sessions = glob(join(ANNO_PATH, "**/*A-ms98-a-trans.text"), recursive=True)
        all_sessions = [
            basename(s)
            .split("-")[0]
            .replace("sw", "")
            .replace("A", "")
            .replace("B", "")
            for s in all_sessions
        ]
        prosody = []
        pbar = tqdm(range(len(all_sessions)))

        # try:
        for split in ["train", "val", "test"]:
            pbar.desc = f"{split.upper()}"

            # Using `datasets_turntaking` to get easy access to switchboard files
            # However they are spread out over 'train', 'val' and 'test' splits
            dset = load_spoken_dialog_audio_dataset(["switchboard"], split=split)
            session_indices = np.array(dset["session"])
            for session in all_sessions:

                if session not in session_indices:
                    continue

                if session in prosody:
                    continue

                # Get data sample
                idx = np.where(session_indices == session)[0].item()
                d = dset[idx]

                # extract prosody
                tmp_prosody = self.extract_prosody_from_session(d, session)
                prosody.append(tmp_prosody)
                pbar.update()
        # except KeyboardInterrupt:
        #     print("Keyboard interrupt saving what's processed -> tmp_prosody.csv")
        #     df = pd.DataFrame(prosody)
        #     df.to_csv("tmp_prosody.csv", index=False)
        #     return df
        makedirs(dirname(savepath), exist_ok=True)
        df = pd.DataFrame(prosody)
        df.to_csv(savepath, index=False)
        print(
            f"Saved PROSODIC global statistics (n={len(df)}/{len(all_sessions)}) -> ",
            savepath,
        )
        return df

    def create_new_filler_prosody_dict(self, filler, pitch, intensity, global_prosody):
        """
        create a new filler-entry with all prosodic information
        """
        # Pad intensity/pitch to same size
        diff = len(pitch) - len(intensity)
        if diff > 0:
            # this seems to match with pitch data
            pre_pad = torch.empty(diff).fill_(intensity[0])
            intensity = torch.cat([pre_pad, intensity])

        new_filler = filler.to_dict()
        #################################################
        # Add utterance prosody
        #################################################
        new_filler["utt_f0_m"] = intensity.mean().item()
        new_filler["utt_f0_s"] = intensity.std().item()
        new_filler["utt_intensity_m"] = pitch[pitch != 0].mean().item()
        new_filler["utt_intensity_s"] = pitch[pitch != 0].std().item()
        #################################################
        # Add filler prosody
        #################################################
        # Find relative start/end of filler in utterance
        fill_rel_start = filler.start - filler.utt_start
        fill_rel_end = filler.end - filler.utt_start
        fill_fs = int(fill_rel_start / self.hop_time) - 1
        fill_fe = int(fill_rel_end / self.hop_time) - 1
        filler_f0 = pitch[fill_fs:fill_fe]
        filler_intensity = intensity[fill_fs:fill_fe]
        new_filler["filler_f0_m"] = filler_f0[filler_f0 != 0].mean().item()
        new_filler["filler_f0_s"] = filler_f0[filler_f0 != 0].std().item()
        new_filler["filler_intensity_m"] = filler_intensity.mean().item()
        new_filler["filler_intensity_s"] = filler_intensity.std().item()
        #################################################
        # Add (global) speaker prosody
        #################################################
        new_filler["speaker_f0_m"] = global_prosody[
            f"{filler.speaker.lower()}_f0_mean"
        ].values.item()
        new_filler["speaker_f0_s"] = global_prosody[
            f"{filler.speaker.lower()}_f0_std"
        ].values.item()
        new_filler["speaker_intensity_m"] = global_prosody[
            f"{filler.speaker.lower()}_intensity_mean"
        ].values.item()
        new_filler["speaker_intensity_s"] = global_prosody[
            f"{filler.speaker.lower()}_intensity_std"
        ].values.item()
        return new_filler

    def extract_filler_prosody(self, filler_path, global_prosody_path):
        if not exists(filler_path):
            print("FILLER path does not exist!")
            print(
                "Please run `python vap_fillers/filler_extraction_from_transcript.py`"
            )
            return

        if not exists(global_prosody_path):
            print("GLOBAL prosody path does not exist!")
            print("Please run `python vap_fillers/filler_prosody.py --aggregate`")
            return

        df = pd.read_csv(filler_path)
        pdf = pd.read_csv(global_prosody_path)
        print("Path: ", filler_path)
        print("N fillers: ", len(df))
        print("path: ", global_prosody_path)
        print("N prosody: ", len(pdf))

        new_filler_df = []
        for row_idx in tqdm(range(len(df)), desc="Extract filler prosody"):
            filler = df.iloc[row_idx]
            global_prosody = pdf[pdf["session"] == filler.session]

            # Load audio of utterance
            audio_path = self.session_to_audio_path(filler.session)
            y, _ = load_waveform(
                audio_path,
                start_time=filler.utt_start,
                end_time=filler.utt_end,
                sample_rate=self.sample_rate,
            )

            # Extract prosody
            pitch, intensity = self.extract_prosody(
                y,
                sample_rate=self.sample_rate,
                hop_time=self.hop_time,
                f0_min=self.f0_min,
                f0_max=self.f0_max,
            )

            new_filler = self.create_new_filler_prosody_dict(
                filler, pitch, intensity, global_prosody
            )
            new_filler_df.append(new_filler)

            # Sanity check
            # I'll probably need it again...
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1, 1)
            # ax.plot(pitch, color="g", label="F0", alpha=0.6)
            # ax.axhline(utt_p_m, color="g", alpha=0.6)
            # ax.axhline(utt_p_m - utt_p_s, color="g", linestyle="dashed", alpha=0.5)
            # ax.axhline(utt_p_m + utt_p_s, color="g", linestyle="dashed", alpha=0.5)
            # ax.plot(intensity, color="b", label="Intensity", alpha=0.6)
            # ax.axhline(utt_i_m, color="b", alpha=0.6)
            # ax.axhline(utt_i_m - utt_i_s, color="b", linestyle="dashed", alpha=0.5)
            # ax.axvline(fill_fs, color="r", linestyle="dashed")
            # ax.axvline(fill_fe, color="r", linestyle="dashed")
            # plt.tight_layout()
            # plt.show()

        fdf = pd.DataFrame(new_filler_df)
        savepath = filler_path.replace(".csv", "_prosody.csv")
        fdf.to_csv(savepath, index=False)
        print("Saved filler with prosody -> ", savepath)
        print(
            f"New/old: {len(fdf)}/{len(df)}",
        )
        return fdf


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-fp", "--filler_path", default="data/FILLER/all_fillers_test.csv"
    )
    parser.add_argument(
        "-gp", "--global_prosody_path", default="data/FILLER/swb_prosody.csv"
    )
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--filler", action="store_true")
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    assert (
        args.aggregate or args.filler
    ), "Must provide either '--aggregate' or '--filler'"

    P = Prosody()
    if args.aggregate:
        prosody_df = P.extract_swb_global_prosody(args.global_prosody_path)

    if args.filler:
        df = P.extract_filler_prosody(args.filler_path, args.global_prosody_path)
