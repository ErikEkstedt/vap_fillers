from argparse import ArgumentParser
from os.path import join
from tqdm import tqdm
import numpy as np
import pandas as pd


SWDA_WORDS = "data/swb_dialog_acts_words"


def load_da_words(filler):
    # Load word dialog acts
    da = pd.read_csv(
        join(SWDA_WORDS, f"sw{filler.session}{filler.speaker}-word-da.csv"),
        names=["utt_idx", "start", "end", "word", "boi", "da", "da_idx"],
    )
    # Find current utterance
    tmp_da = da[da["utt_idx"] == filler.utt_idx]
    return tmp_da


def append_dialog_acts(df):
    new_df = []
    for row_idx in tqdm(range(len(df)), desc="Append Dialog Acts"):
        filler = df.iloc[row_idx]
        tmp_da = load_da_words(filler)
        # location in da (same as utterance loc)
        loc_in_da = np.where(tmp_da["start"] == filler.start)[0].item()
        if not len(tmp_da) == filler.utt_n_words:
            print("len(da) != len(filler.utt)")
        if not loc_in_da == filler.utt_loc:
            print("loc_in_da != filler.utt_loc")
        # Create new filler
        new_filler = filler.to_dict()
        new_filler["da"] = tmp_da.iloc[loc_in_da]["da"]
        new_df.append(new_filler)
    return pd.DataFrame(new_df)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--filler", type=str, default="results/all_fillers_test_prosody_model.csv"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.filler)
    new_df = append_dialog_acts(df)
    savepath = args.filler.replace(".csv", "_da.csv")
    new_df.to_csv(savepath)
    print("Added Dialog Acts and saved -> ", savepath)
