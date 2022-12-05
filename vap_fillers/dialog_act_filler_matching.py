from os.path import basename, join
from os import makedirs
from glob import glob
from tqdm import tqdm
from vap.utils import read_txt, read_json, write_json, write_txt
import re
import numpy as np
import matplotlib.pyplot as plt

TEST_FILLER_PATH = "data/test_fillers.txt"
FILLER_DAL_PATH = "data/test_fillers_dal.txt"
OUTPUT_PATH = "data/test_fillers_with_da.txt"
SWDA_W_PATH = "data/swb_dialog_acts_words"
SWDA_U_PATH = "data/swb_dialog_acts_utterances"

# SHOW DATA
COLUMNS = ["session", "start", "end", "speaker", "uh_or_um", "loc", "words_in_da", "da"]
cols2idx = {k: v for v, k in enumerate(COLUMNS)}
idx2cols = {k: v for v, k in cols2idx.items()}


def read_speaker_word_da(path):
    def row_to_segment(row):
        """
        FILES: e.g. "data/swb_da/sw2095A-word-da.csv"

        includes these rows:
        sw2095A-ms98-a-0001,0.790000,0.910000,all,B,qy,u-1-1
        sw2095A-ms98-a-0001,0.910000,1.080000,right,I,qy,u-1-1
        sw2095A-ms98-a-0003,5.108875,5.710750,okay,B,bk,u-3-1
        sw2095A-ms98-a-0005,11.360625,11.700625,yeah,B,aa,u-5-1
        sw2095A-ms98-a-0005,11.700625,11.840625,it's,O,,
        sw2095A-ms98-a-0005,11.840625,12.050625,it,B,sv,u-5-2
        sw2095A-ms98-a-0005,12.050625,12.310625,it's,I,sv,u-5-2
        ...


        utterance_id:   str, Id of utterance in transcription
        start:          float, start time of word
        end:            float, end time of word
        word:           str, the text of the word
        boi:            str, "B": beginning of utterance, "I": inside an utterance, "O": outside? i don't really know...
        da:             str, dialog act e.g. "qy" - yes/no question. see DAS
        da_idx:
        """

        # BREAKS
        # sw2751A-ms98-a-0031,99.761125,99.971125,they're,B,"sd,o@",u-37-2
        # sw2713A-ms98-a-0052,234.395000,234.473875,go,B,"ad,qy@",u-55-1 -> '"(sd),(o@)"'
        # sw2996B-ms98-a-0020,118.435375,118.595375,i,B,"na,sd,o@",u-31-1 -> '"(na),(sd),(o@)"'
        if '"' in row:
            row = re.sub(r'"(.+?),(.+?),(.+?)"', r"\1-\2", row)
            row = re.sub(r'"(.+?),(.+?)"', r"\1-\2", row)

        utt_id, start, end, word, boi, da, da_idx = row.split(",")

        start = float(start)
        end = float(end)
        return {
            "word": word,
            "start": start,
            "end": end,
            "boi": boi,
            "da": da,
            "da_idx": da_idx,
            "utt_id": utt_id,
        }

    data = []
    for row in read_txt(path):
        data.append(row_to_segment(row))
    return data


def dialog_act_utterances():
    def segments_to_utt(segments):
        utt_id = segments[0]["utt_id"]
        session = utt_id[2:6]
        speaker = utt_id[6:7]

        utt = {
            "start": segments[0]["start"],
            "end": segments[-1]["end"],
            "words": [],
            "starts": [],
            "ends": [],
            "boi": [],
            "da": segments[0]["da"],
            "da_idx": segments[0]["da_idx"],
            "session": session,
            "speaker": speaker,
        }
        text = ""
        for segment in segments:
            text += " " + segment["word"]
            utt["words"].append(segment["word"])
            utt["starts"].append(segment["start"])
            utt["ends"].append(segment["end"])
            utt["boi"].append(segment["boi"])
        utt["text"] = text.strip()
        return utt

    makedirs(SWDA_U_PATH, exist_ok=True)

    filepaths = glob(join(SWDA_W_PATH, "**/*.csv"), recursive=True)
    data = {}
    for filepath in tqdm(filepaths):
        word_info = read_speaker_word_da(filepath)

        current_words = [word_info[0]]
        last_da = word_info[0]["da"]

        utterances = []
        for word in word_info[1:]:
            if word["boi"] != "B":
                current_words.append(word)
            else:
                if last_da not in data:
                    data[last_da] = []

                # Add previous words as utterance
                utterances.append(segments_to_utt(current_words))
                current_words = [word]  # Update

        # Add last
        utterances.append(segments_to_utt(current_words))

        filename = join(SWDA_U_PATH, basename(filepath).replace(".csv", ".json"))
        write_json(utterances, filename)


def find_filler_pos_rel_da(session, speaker, filler_start, filler_end):
    if isinstance(speaker, int):
        speaker = "A" if speaker == 0 else "B"
    utterances = read_json(join(SWDA_U_PATH, f"sw{session}{speaker}-word-da.json"))
    loc = None
    words_in_da = None
    da = None
    done = False
    for utt in utterances:
        # is filler in this utt?
        if utt["start"] <= filler_start <= utt["end"]:
            n = len(utt["starts"])
            for idx in range(n):
                if (
                    utt["starts"][idx] <= filler_start <= utt["ends"][idx]
                    or utt["starts"][idx] <= filler_end <= utt["ends"][idx]
                ):
                    # found filler. Where is it in utterance?
                    loc = idx
                    words_in_da = n
                    da = utt["da"]
        if done:
            break
    return loc, words_in_da, da


def add_dialog_act_info_to_fillers():
    fillers = read_txt(TEST_FILLER_PATH)
    fillers_with_dal = []
    for filler in tqdm(fillers):
        session, start, end, speaker, uh_or_um = filler.split()

        filler_start = float(start)
        filler_end = float(end)
        utterances = read_json(join(SWDA_U_PATH, f"sw{session}{speaker}-word-da.json"))
        speaker = 0 if speaker == "A" else 1

        da, loc, words_in_da = find_filler_pos_rel_da(filler_start, utterances)

        row = f"{session},{start},{end},{speaker},{uh_or_um},{da},{loc},{words_in_da}"
        if loc is None:
            print("Broken: ")
            print(filler)
            print(row)
            input()
        fillers_with_dal.append(row)
    write_txt(fillers_with_dal, FILLER_DAL_PATH)
    return fillers_with_dal


def plot_filler_da_loc_distribution(fillers):
    uhs = np.where(fillers[:, cols2idx["uh_or_um"]] == "uh")[0]
    ums = np.where(fillers[:, cols2idx["uh_or_um"]] == "um")[0]
    uhloc = np.stack(
        (
            fillers[uhs, cols2idx["loc"]].astype(float),
            fillers[uhs, cols2idx["words_in_da"]].astype(float),
        )
    )
    umloc = np.stack(
        (
            fillers[ums, cols2idx["loc"]].astype(float),
            fillers[ums, cols2idx["words_in_da"]].astype(float),
        )
    )

    ruh = uhloc[0] / uhloc[1]
    rum = umloc[0] / umloc[1]

    nuh, bins = np.histogram(ruh, range=(0, 1), bins=10)
    num, bins = np.histogram(rum, range=(0, 1), bins=10)
    nuh = nuh / nuh.sum()
    num = num / num.sum()

    x = bins[:-1] + (bins[1] - bins[0]) / 2
    w = x[1] - x[0]

    fig, ax = plt.subplots(1, 1)
    ax.bar(x, (nuh * 100).round(1), width=w, color="b", alpha=0.5, label="UH")
    ax.bar(x, (num * 100).round(1), width=w, color="r", alpha=0.5, label="UM")
    # ax.hist(ruh, color='b', alpha=0.6, range=(0, 1), bins=10)
    # ax.hist(rum, color='r', alpha=0.6, range=(0, 1), bins=10)
    ax.set_ylabel("%")
    ax.set_xlabel("word-location to number of words")
    ax.legend(loc="upper right")
    return fig, ax


if __name__ == "__main__":

    # dialog_act_utterances()
    # fillers = add_dialog_act_info_to_fillers()

    fillers = read_txt(FILLER_DAL_PATH)
    fillers = np.array([f.split(",") for f in read_txt(FILLER_DAL_PATH)])
    fig, ax = plot_filler_da_loc_distribution(fillers)
    plt.show()
    locs = {"uh": [], "um": []}
