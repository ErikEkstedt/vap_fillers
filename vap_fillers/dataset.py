from os.path import basename, dirname, join, expanduser
from pathlib import Path
from glob import glob
from tqdm import tqdm
import random
import re
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from vap.audio import load_waveform, get_audio_info
from vap.utils import read_txt, read_json

"""
https://www.isip.piconepress.com/projects/switchboard/doc/transcription_guidelines/transcription_guidelines.pdf

swb_ms98_transcriptions/
    .
    ├── 20
    ├── 21
    ├── ...
    └── 49
        ├── 4901
        ├── ...
        └── 4940
            ├── sw4940A-ms98-a-trans.text
            ├── sw4940A-ms98-a-word.text
            ├── sw4940B-ms98-a-trans.text
            └── sw4940B-ms98-a-word.text
"""

ANNO_PATH = join(
    expanduser("~"), "projects", "data/switchboard/annotations/swb_ms98_transcriptions"
)
AUDIO_PATH = join(expanduser("~"), "projects/data/switchboard/audio")
AUDIO_REL_PATH = "data/relative_audio_path.json"

LAUGHTER = ["[laughter]"]
HESITATION = ["uh", "ah", "um", "hm"]
YES_SOUNDS = ["uh-huh", "um-hum", "yeah", "yep"]
NO_SOUNDS = ["huh-uh", "hum-um", "nope"]
YES_NO_SOUNDS = YES_SOUNDS + NO_SOUNDS


# TODO: file:///home/erik/Downloads/thesis.pdf
# TODO: extract backchannels that follow [silence]
# TODO: Follow the thesis and only train on BC/NON-BC-Areas, but include more context.


def contains_laughter(s):
    if re.search(r"\[laughter", s):
        return True
    return False


def contains_hesitation(s):
    words = s.split()
    for h in HESITATION:
        if h in words:
            return True
    return False


def contains_yes_no_sound(s):
    words = s.split()
    for h in YES_NO_SOUNDS:
        if h in words:
            return True
    return False


def get_event(utt_info, condition):
    if condition == "laughter":
        cond = contains_laughter
    elif condition == "hesitation":
        cond = contains_hesitation
    elif condition == "yes_no_sound":
        cond = contains_yes_no_sound
    else:
        raise NotImplementedError(
            f'{condition} not implemented. Try ["laughter", "hesitation", "yes_no_sound"]'
        )

    in_utt = []
    for ii, w in enumerate(utt_info["words"]):
        if cond(w):
            in_utt.append(
                {
                    "session": utt_info["session"],
                    "speaker": utt_info["speaker"],
                    "utt_idx": utt_info["utt_idx"],
                    "text": w,
                    "start": utt_info["starts"][ii],
                    "end": utt_info["ends"][ii],
                    "event": condition,
                    "event_word_loc_in_utt": ii,
                    "n_words_in_utt": len(utt_info["words"]),
                }
            )
    return in_utt


def plot_basic_stats(df, title, plot=False):
    duration = df["end"] - df["start"]
    r = df["event_word_loc_in_utt"] / df["n_words_in_utt"]
    fig, ax = plt.subplots(2, 1)
    fig.suptitle(title)
    duration.hist(bins=100, ax=ax[0], label="duration")
    r.hist(bins=100, ax=ax[1], label="relative placement in utterance")
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig


class SWBExtractor:
    def __init__(
        self,
        anno_path=ANNO_PATH,
    ):
        self.anno_path = anno_path
        self.session_rel_path = self._session_rel_paths()

    def _session_rel_paths(self):
        return {
            basename(dirname(fp).replace(self.anno_path, "")): dirname(fp).replace(
                self.anno_path + "/", ""
            )
            for fp in glob(
                join(self.anno_path, "**/*A-ms98-a-trans.text"), recursive=True
            )
        }

    def get_filename(self, session, speaker, option, full_path=True):
        if isinstance(speaker, int):
            speaker = "A" if speaker == 0 else "B"

        filename = f"sw{session}{speaker}-ms98-a-"

        if option in ["word", "words"]:
            filename += "word.text"
        else:
            filename += "trans.text"

        if full_path:
            filename = join(self.anno_path, self.session_rel_path[session], filename)
        return filename

    def read_utter_trans(self, path, session, speaker):
        """extract utterance annotation"""
        # trans = []
        trans = {}
        for row in read_txt(path):
            utt_idx, start, end, *text = row.split(" ")
            text = " ".join(text)
            start = float(start)
            end = float(end)
            if text == "[silence]":
                continue
            if text == "[noise]":
                continue
            trans[utt_idx] = {
                "session": session,
                "speaker": speaker,
                "utt_idx": utt_idx,
                "start": start,
                "end": end,
                "text": text,
            }
        return trans

    def read_word_trans(self, path):
        trans = []
        for row in read_txt(path):
            utt_idx, start, end, text = row.strip().split()
            start = float(start)
            end = float(end)
            if text == "[silence]":
                continue
            if text == "[noise]":
                continue
            trans.append({"utt_idx": utt_idx, "start": start, "end": end, "text": text})
        return trans

    def combine_utterance_and_words(self, utterances, words):
        utters = {}
        for utt_idx, utterance in utterances.items():
            utters[utt_idx] = utterance
            utters[utt_idx]["words"] = []
            utters[utt_idx]["starts"] = []
            utters[utt_idx]["ends"] = []
            for w in words:
                # words are after the utterance ends
                if utterance["end"] < w["start"]:
                    break

                if w["utt_idx"] == utt_idx:
                    utters[utt_idx]["words"].append(w["text"])
                    utters[utt_idx]["starts"].append(w["start"])
                    utters[utt_idx]["ends"].append(w["end"])

        return utters

    def extract_session_data(self, session):
        A_utters = self.read_utter_trans(
            self.get_filename(session, "A", "utter"), session, "A"
        )
        A_words = self.read_word_trans(self.get_filename(session, "A", "word"))

        B_utters = self.read_utter_trans(
            self.get_filename(session, "B", "utter"), session, "B"
        )
        B_words = self.read_word_trans(self.get_filename(session, "B", "word"))
        return {
            "A": self.combine_utterance_and_words(A_utters, A_words),
            "B": self.combine_utterance_and_words(B_utters, B_words),
        }

    def is_other_too_close(self, w, words_other, min_duration_to_other):
        is_too_close = False
        for wo in words_other:
            if wo["end"] < w["start"] - min_duration_to_other:
                # We are early before the "safe" distance of the filler
                continue

            if wo["start"] > w["end"] + min_duration_to_other:
                # we are after the "safe" distance of the filler
                break

            pre_limit = w["start"] - min_duration_to_other
            post_limit = w["end"] + min_duration_to_other

            # word starts in NONO-REGION
            if pre_limit <= wo["start"] <= post_limit:
                is_too_close = True
                break

            # word ends in NONO-REGION
            if pre_limit <= wo["end"] <= post_limit:
                is_too_close = True
                break

            # This should never happen but..
            # the word is longer than the NONO-REGION
            if wo["start"] < pre_limit and post_limit < wo["end"]:
                is_too_close = True
                break

        return is_too_close

    def extract_events(self):
        laughter = []
        hesitation = []
        yes_no_sound = []

        sessions = list(self.session_rel_path.keys())
        for session in tqdm(sessions):
            info = self.extract_session_data(session)
            for speaker in ["A", "B"]:
                for _, utt_info in info[speaker].items():
                    text = utt_info["text"]
                    if contains_laughter(text):
                        laughter += get_event(utt_info, "laughter")
                    if contains_hesitation(text):
                        hesitation += get_event(utt_info, "hesitation")
                    if contains_yes_no_sound(text):
                        yes_no_sound += get_event(utt_info, "yes_no_sound")

        laughter = pd.DataFrame(laughter)
        hesitation = pd.DataFrame(hesitation)
        yes_no_sound = pd.DataFrame(yes_no_sound)

        return laughter, hesitation, yes_no_sound

    def save_events(self, savedir="results/events", split_root="data/splits"):
        laughter, hesitation, yes_no_sound = self.extract_events()

        Path(savedir).mkdir(parents=True, exist_ok=True)

        laughter_path = join(savedir, "laughter.csv")
        hesitation_path = join(savedir, "hesitation.csv")
        yes_no_sound_path = join(savedir, "yes_no_sound.csv")
        laughter.to_csv(laughter_path, index=False)
        hesitation.to_csv(hesitation_path, index=False)
        yes_no_sound.to_csv(yes_no_sound_path, index=False)
        print("Saved Laughter -> ", laughter_path)
        print("Saved Hesitations -> ", hesitation_path)
        print("Saved yes_no_sound -> ", yes_no_sound_path)
        print("-" * 30)

        # Create Splits
        for split in ["train", "val", "test"]:
            sessions = read_txt(f"{split_root}/{split}.txt")
            split_laughter = laughter[laughter["session"].isin(sessions)]
            split_hesitation = hesitation[hesitation["session"].isin(sessions)]
            split_yes_no_sound = yes_no_sound[yes_no_sound["session"].isin(sessions)]
            split_laughter.to_csv(
                laughter_path.replace(".csv", f"_{split}.csv"), index=False
            )
            split_hesitation.to_csv(
                hesitation_path.replace(".csv", f"_{split}.csv"), index=False
            )
            split_yes_no_sound.to_csv(
                yes_no_sound_path.replace(".csv", f"_{split}.csv"), index=False
            )
            print(
                f"{split} Laughter -> ", laughter_path.replace(".csv", f"_{split}.csv")
            )
            print(
                f"{split} Hesitations -> ",
                hesitation_path.replace(".csv", f"_{split}.csv"),
            )
            print(
                f"{split} yes_no_sound -> ",
                yes_no_sound_path.replace(".csv", f"_{split}.csv"),
            )

        return laughter, hesitation, yes_no_sound


class SpeechEventDataset(Dataset):
    def __init__(
        self,
        path,
        window_duration=0.5,
        context_min=3,
        duration=10,
        audio_path=AUDIO_PATH,
        audio_rel_path=AUDIO_REL_PATH,
        sample_rate=16_000,
        frame_hz=50,
    ):
        self.path = path
        self.df = pd.read_csv(path)
        self.audio_path = audio_path
        self.audio_rel_paths = read_json(audio_rel_path)

        self.window_duration = window_duration
        self.window_frames = int(window_duration * frame_hz)
        self.duration = duration
        self.context_min = context_min
        self.sample_rate = sample_rate
        self.n_samples = int(sample_rate * duration)
        self.frame_hz = frame_hz

    def __len__(self):
        return len(self.df)

    def labels_to_onehot(self, labels):
        n_frames = int(self.duration * self.frame_hz)
        oh = torch.zeros((n_frames, 2))
        for speaker, start, end in labels:
            start_frame = int(start * self.frame_hz) - self.window_frames
            if start_frame < 0:
                start_frame = 0
            end_frame = round(end * self.frame_hz)
            oh[start_frame : end_frame + 1, speaker] = 1
        return oh

    def get_sample(self, df, idx, context_min, duration, wav_duration):
        row = df.iloc[idx]
        speaker_idx = 0 if row.speaker == "A" else 1

        min_limit = row.end - duration
        max_limit = row.start - context_min

        # If we try to sample before the audio starts (negative values)
        # we simply sample somewhere between the start of the session and the label.start
        if min_limit < 0:
            min_limit = 0
            if max_limit < 0:
                max_limit = row.start

        # Sample start point
        r = random.random()
        start_time = min_limit + r * (max_limit - min_limit)
        end_time = start_time + duration

        # if we try to go beyond the end
        # we simply translate the sample to fit inside the session
        # i.e. we sample the last `duration` seconds of the audio
        if end_time > wav_duration:
            diff = end_time - wav_duration
            end_time = wav_duration
            start_time -= diff

        # Negative start time

        # Add the label for the focus point
        labels = [[speaker_idx, row.start - start_time, row.end - start_time]]

        # Find other labels inside of the sampled region
        # session_events: se
        se = df[df["session"] == row.session]
        start_after = se[se["start"] >= start_time]
        end_before = start_after[start_after["end"] <= end_time]
        others = end_before[end_before["start"] != row.start]
        if len(others) > 0:
            for jj in range(len(others)):
                other = others.iloc[jj]
                other_speaker_idx = 0 if other.speaker == "A" else 1
                labels.append(
                    [
                        other_speaker_idx,
                        other.start - start_time,
                        other.end - start_time,
                    ]
                )
        return labels, str(row.session), start_time, end_time

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        session = str(row.session)
        audio_path = join(self.audio_path, self.audio_rel_paths[session] + ".wav")
        wav_duration = get_audio_info(audio_path)["duration"]

        labels, session, start, end = self.get_sample(
            self.df,
            idx,
            context_min=self.context_min,
            duration=self.duration,
            wav_duration=wav_duration,
        )
        waveform, _ = load_waveform(
            audio_path,
            start_time=start,
            end_time=end,
            sample_rate=self.sample_rate,
        )
        waveform = waveform[..., : self.n_samples]

        onehot_label = self.labels_to_onehot(labels)
        # add batch dim
        # onehot_label = onehot_label.unsqueeze(0)
        # waveform = waveform.unsqueeze(0)
        return {"session": session, "waveform": waveform, "label": onehot_label}


if __name__ == "__main__":

    # e = SWBExtractor()
    # laughter, hesitation, yes_no_sound = e.save_events()
    # laughter = pd.read_csv("results/events/laughter.csv")
    # hesitation = pd.read_csv("results/events/hesitation.csv")
    # yes_no_sound = pd.read_csv("results/events/yes_no_sound.csv")
    # laugh_fig = plot_basic_stats(laughter, "Laughter", True)
    # hesitation_fig = plot_basic_stats(hesitation, "Hesitation", True)
    # yes_no_fig = plot_basic_stats(yes_no_sound, "Yes-No-Sound", True)

    dset = SpeechEventDataset(
        # path="results/events/hesitation_val.csv",
        path="results/events/laughter_val.csv",
        # path="results/events/yes_no_sound_val.csv",
        window_duration=0.5,
        context_min=3,
        duration=10,
    )
    dloader = DataLoader(dset, batch_size=4, num_workers=4)

    batch = next(iter(dloader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    print(f"DSET {dset.path} N: {len(dset)}")
    d = dset[0]

    x = d["waveform"]
    y = d["label"]
    print("x: ", tuple(x.shape))
    print("y: ", tuple(y.shape))
