"""
When extracting fillers we also need to know about the local context

- IPU-TURN-Location
    - Is the backchannel occuring at the beginning, middle, end of a "turn"
- Local utterance
    - What's the prosodic signature of the local utterance
    - Does the prosodic signature of the filler stick out?
    - Prosody extraction: extracting prosody over longer duration may give
      better pitch/intensity estimation 

-----------------------------------------------------------------------------
"""


from os.path import basename, join
from glob import glob
from tqdm import tqdm
import pandas as pd

from vap.utils import read_json, read_txt

F0_PATH = "data/pitch_information_test.csv"
INTENSITY_PATH = "data/intensity_information_test.csv"
ANNO_PATH = "../../data/switchboard/annotations/swb_ms98_transcriptions"
FILLER = ["uh", "um"]


def words_to_vad_list(words, min_diff=0.05):
    vad_list = []
    current_segment = [words[0]["start"], words[0]["end"]]
    for winfo in words[1:]:
        if winfo["start"] - current_segment[-1] <= min_diff:
            current_segment[-1] = winfo["end"]
        else:
            vad_list.append(current_segment)
            current_segment = [winfo["start"], winfo["end"]]
    vad_list.append(current_segment)
    return vad_list


def _extract_global_speaker_session_pitch(savepath="data/all_fillers_f0.csv"):
    from datasets_turntaking.dialog_audio_dataset import (
        load_spoken_dialog_audio_dataset,
    )
    import numpy as np
    import torch
    from vap.audio import load_waveform
    from vap.functional import pitch_praat

    REL_PATH = "data/relative_audio_path.json"
    TEST_FILE_PATH = "data/test.txt"
    AUDIO_ROOT = join("../../data/switchboard/audio")
    SAMPLE_RATE = 16_000

    def extract_pitch_from_session(session, min_chunk_time=0.1):
        min_samples = min_chunk_time * SAMPLE_RATE

        idx = np.where(sess_idx == session)[0].item()
        d = dset[idx]

        audio_path = join(AUDIO_ROOT, session_to_rel_path[session] + ".wav")
        waveform, _ = load_waveform(audio_path, sample_rate=SAMPLE_RATE)

        session_pitch = {}
        # pitch[session] = {}
        for speaker in ["A", "B"]:
            channel = 0 if speaker == "A" else 1
            vad_list = torch.tensor(d["vad_list"][channel])

            tmp_pitch = []
            for start_time, end_time in vad_list:
                start = (start_time * SAMPLE_RATE).long()
                end = (end_time * SAMPLE_RATE).long()
                if end - start < min_samples:
                    continue
                y = waveform[channel, start:end]
                try:
                    p = pitch_praat(y, sample_rate=SAMPLE_RATE)
                except:
                    print("y: ", tuple(y.shape))
                    assert False
                tmp_pitch.append(p[p != 0])
            pp = torch.cat(tmp_pitch)
            session_pitch[speaker] = {
                "mean": pp.mean().item(),
                "std": pp.std().item(),
            }
        return session_pitch

    all_sessions = glob(join(ANNO_PATH, "**/*A-ms98-a-trans.text"), recursive=True)
    all_sessions = [
        basename(s).split("-")[0].replace("sw", "").replace("A", "").replace("B", "")
        for s in all_sessions
    ]
    session_to_rel_path = read_json(REL_PATH)

    pitch_data = []
    try:
        pbar = tqdm(range(len(all_sessions)))
        for split in ["train", "val", "test"]:
            pbar.desc = f"{split.upper()}"
            dset = load_spoken_dialog_audio_dataset(
                ["switchboard"], split=split, min_word_vad_diff=0.1
            )
            sess_idx = np.array(dset["session"])
            for session in all_sessions:
                if session not in sess_idx:
                    continue
                if session not in pitch_data:
                    pitch = extract_pitch_from_session(session, min_chunk_time=0.2)
                    pitch_data.append(
                        {
                            "session": session,
                            "a_mean": pitch["A"]["mean"],
                            "a_std": pitch["A"]["std"],
                            "b_mean": pitch["B"]["mean"],
                            "b_std": pitch["B"]["std"],
                        }
                    )
                    pbar.update()
    except Exception as e:
        print(e)

    df = pd.DataFrame(pitch_data)
    df.to_csv(savepath, index=False)
    print(f"Saved F0 global stats (n={len(df)}/{len(all_sessions)}) -> ", savepath)
    return df


class FillerExtractor:
    def __init__(
        self, anno_path=ANNO_PATH, min_pause_after_filler=0.2, min_duration_to_other=1.0
    ):
        self.anno_path = anno_path
        self.min_pause_after_filler = min_pause_after_filler
        self.min_duration_to_other = min_duration_to_other

    def read_utter_trans(self, path):
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
            # trans.append({"utt_idx": utt_idx, "start": start, "end": end, "text": text})
            trans[utt_idx] = {"start": start, "end": end, "text": text}
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

    def combine_utterance_and_words(self, speaker_info):
        words = speaker_info["word"]
        utters = speaker_info["utter"]
        utterances = {}
        for utt_idx, utterance in utters.items():
            utterances[utt_idx] = utterance
            utterances[utt_idx]["words"] = []  # {'word': [], 'starts': [], 'ends': []}
            utterances[utt_idx]["starts"] = []
            utterances[utt_idx]["ends"] = []
            for w in words:
                # words are after the utterance ends
                if utterance["end"] < w["start"]:
                    break

                if w["utt_idx"] == utt_idx:
                    utterances[utt_idx]["words"].append(w["text"])
                    utterances[utt_idx]["starts"].append(w["start"])
                    utterances[utt_idx]["ends"].append(w["end"])

        return utterances

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

    def filler_extraction(self, min_pause_after_filler=0.2, min_duration_to_other=1.0):
        """
        Extracts fillers (see `FILLER`) from word/utterance annotation

        creates a pandas.DataFrame and save to csv

        colums:
        Index(['session', 'speaker', 'filler', 'start', 'end', 'utt_idx', 'utt_start',
               'utt_end', 'utt_n_words'],
              dtype='object')
        """
        # min_pause_after_filler = 0.2
        # min_duration_to_other = 1
        filler_data = []
        files = glob(join(self.anno_path, "**/*A-ms98-a-trans.text"), recursive=True)
        files.sort()
        for file in tqdm(files):
            name = basename(file)
            # Its slower to not do everyting in one loop
            # but it's simpler to track... i hope...
            trans = {
                "A": {
                    "utter": self.read_utter_trans(file),
                    "word": self.read_word_trans(file.replace("-trans", "-word")),
                },
                "B": {
                    "utter": self.read_utter_trans(
                        file.replace("A-ms98-a", "B-ms98-a")
                    ),
                    "word": self.read_word_trans(
                        file.replace("A-ms98-a-trans", "B-ms98-a-word")
                    ),
                },
            }
            info = {
                "A": self.combine_utterance_and_words(trans["A"]),
                "B": self.combine_utterance_and_words(trans["B"]),
            }
            session = (
                name.split("-")[0].replace("A", "").replace("B", "").replace("sw", "")
            )
            # Find the fillers
            for speaker, other_speaker in zip(["A", "B"], ["B", "A"]):
                words = trans[speaker]["word"]
                words_other = trans[other_speaker]["word"]
                for w, wnext in zip(words[:-1], words[1:]):
                    if not w["text"] in FILLER:
                        continue
                    # Intra speaker pause
                    pause_after_filler = wnext["start"] - w["end"]
                    if pause_after_filler < min_pause_after_filler:
                        continue
                    # Inter speaker pause
                    # is the other speaker active very close to this filler?
                    other_is_close = self.is_other_too_close(
                        w, words_other, min_duration_to_other=min_duration_to_other
                    )
                    if other_is_close:
                        continue
                    ##################################
                    # FILLER IS OKAY!
                    ##################################
                    # Do someting with the filler...
                    utt_idx = w["utt_idx"]
                    utt = info[speaker][utt_idx]
                    filler_rel_utter_location = -1
                    for ii, s in enumerate(utt["starts"]):
                        if s == w["start"]:
                            filler_rel_utter_location = ii
                            break
                    if filler_rel_utter_location == -1:
                        print("This should not happen....")
                        print("filler not in utterances?")
                        print(w)
                        for word, s, e in zip(utt["words"], utt["starts"], utt["ends"]):
                            print(word, s, e)
                    # pm = f0_df[]
                    filler_data.append(
                        {
                            "session": session,
                            "speaker": speaker,
                            "filler": w["text"],
                            "start": w["start"],
                            "end": w["end"],
                            "utt_idx": utt_idx,
                            "utt_start": utt["start"],
                            "utt_end": utt["end"],
                            "utt_n_words": len(utt["words"]),
                        }
                    )
        return pd.DataFrame(filler_data)

    def extract(self):
        return self.filler_extraction()


if __name__ == "__main__":

    # Extract global F0 from all sessions
    df = _extract_global_speaker_session_pitch()

    # change format of older global F0 etc
    # _ = _json_to_dataframe_csv(F0_PATH)
    # _ = _json_to_dataframe_csv(INTENSITY_PATH)

    # Must extract all prosody
    f0_df = pd.read_csv(F0_PATH)
    # intensity_df = pd.read_csv(INTENSITY_PATH)

    # df = FillerExtractor().extract()
    # df.to_csv("data/all_fillers.csv", index=False)
    fdf = pd.read_csv("data/all_fillers_f0.csv")
