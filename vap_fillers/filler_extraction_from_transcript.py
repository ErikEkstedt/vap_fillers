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
from os import makedirs
from glob import glob
from tqdm import tqdm
import pandas as pd
from vap.utils import read_txt, write_json

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


class FillerExtractor:
    def __init__(
        self,
        root,
        anno_path=ANNO_PATH,
        min_pause_after_filler=0.2,
        min_duration_to_other=1.0,
    ):
        self.root = root
        self.anno_path = anno_path
        self.min_pause_after_filler = min_pause_after_filler
        self.min_duration_to_other = min_duration_to_other
        self.total_is_too_close = 0

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

        makedirs(self.root, exist_ok=True)
        trans_path = join(self.root, "transcripts")
        makedirs(trans_path, exist_ok=True)

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
            write_json(info, join(trans_path, session + ".json"))
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
                        self.total_is_too_close += 1
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
                            "utt_loc": filler_rel_utter_location,
                        }
                    )
        return pd.DataFrame(filler_data)

    def extract(self):

        df = self.filler_extraction(
            self.min_pause_after_filler, self.min_duration_to_other
        )
        savepath = join(self.root, "all_fillers.csv")
        df.to_csv(savepath, index=False)
        print(f"Total fillers: {len(df)}")
        print(f"Omitted {F.total_is_too_close} entries too close to other speaker")
        print("Saved -> ", savepath)
        return df, savepath


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--root", default="data/FILLER")
    parser.add_argument("--train_files", default="data/FILLER/splits/train.txt")
    parser.add_argument("--val_files", default="data/FILLER/splits/val.txt")
    parser.add_argument("--test_files", default="data/FILLER/splits/test.txt")
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    F = FillerExtractor(args.root)
    df, savepath = F.extract()

    # Is this not working in the script????
    # split
    # savepath = "data/FILLER/all_fillers.csv"
    savepath = join(args.root, "all_fillers.csv")
    # df = pd.read_csv(savepath)
    # print(len(df))

    test = [int(f) for f in read_txt(args.test_files)]
    train = [int(f) for f in read_txt(args.train_files)]
    val = [int(f) for f in read_txt(args.val_files)]
    print("train_files: ", len(train))
    print("val_files: ", len(val))
    print("test_files: ", len(test))

    train_df = df[df["session"].isin(train)]
    val_df = df[df["session"].isin(val)]
    test_df = df[df["session"].isin(test)]

    train_df.to_csv(savepath.replace(".csv", "_train.csv"), index=False)
    val_df.to_csv(savepath.replace(".csv", "_val.csv"), index=False)
    test_df.to_csv(savepath.replace(".csv", "_test.csv"), index=False)
    print(f"Train Saved {len(train_df)}")
    print(f"Val Saved {len(val_df)}")
    print(f"Test Saved {len(test_df)}")
    print("Sum: ", len(test_df) + len(val_df) + len(train_df))
    print("Total: ", len(df))
