from os.path import join, basename, dirname
import glob
from vap.utils import write_txt


DIRPATH = "data/where_is_the_hesitation/Stimuli"


"""
* Where are the fillers
    - word time align
* Model entropy conntected to label certainty?
* HOLD probs in silence after filler 
* Generate question but insert filler at the end
    - conditional filler
        - how does the tts change the infromation in the question given that they produce a filler at the end?
        - SHIFT decreases at question boundary if generating filler?
    - unconditional filler
        - generate the question, generate the question+filler, steal filler from latter and put in former
"""


# # Listening to the samples I guessed the filler (eh, um) which fits almost all samples
# # these are then save to the same name as the wav-files.
# # I.e. accurate_0_0_initial.txt, accurate_0_0_initial.wav
# # This is 'required' to run Montreal Forced Aligner

# with 'eh' fillers. Which don't seem to be properly recognized by the forced aligner
# name_to_text = {
#     "accurate": {
#         "no_filler": "I think that's the more accurate version",
#         "initial_filler": "eh I think that's the more accurate version",
#         "medial_filler": "I think that's the more eh accurate version",
#     },
#     "bank": {
#         "no_filler": "I think I've got about three hundred dollars in my bank account",
#         "initial_filler": "eh I think I've got about three hundred dollars in my bank account",
#         "medial_filler": "I think I've got about eh three hundred dollars in my bank account",
#     },
#     "heart": {
#         "no_filler": "I think it's time you start listening to your heart",
#         "initial_filler": "eh I think it's time you start listening to your heart",
#         "medial_filler": "I think it's time you eh start listening to your heart",
#     },
#     "kegs": {
#         "no_filler": "I think the kegs are upstairs",
#         "initial_filler": "eh I think the kegs are upstairs",
#         "medial_filler": "I think the eh kegs are upstairs",
#     },
#     "misunderstandings": {
#         "no_filler": "I think we've had a series of major misunderstandings",
#         "initial_filler": "eh I think we've had a series of major misunderstandings",
#         "medial_filler": "I think we've had a series of major eh misunderstandings",
#     },
#     "refund": {
#         "no_filler": "I think you should ask for a refund",
#         "initial_filler": "um I think you should ask for a refund",
#         "medial_filler": "I think you should ask for um a refund",
#     },
#     "simon": {
#         "no_filler": "I think it was simon they were after",
#         "initial_filler": "uh I think it was simon they were after",
#         "medial_filler": "I think it was uh simon they were after",
#     },
#     "work": {
#         "no_filler": "I think it's gonna work out alright this time",
#         "initial_filler": "uh I think it's gonna work out alright this time",
#         "medial_filler": "I think it's gonna work out uh alright this time",
#     },
# }

# with 'eh' replaced with 'uh'
name_to_text = {
    "accurate": {
        "no_filler": "I think that's the more accurate version",
        "initial_filler": "uh I think that's the more accurate version",
        "medial_filler": "I think that's the more uh accurate version",
    },
    "bank": {
        "no_filler": "I think I've got about three hundred dollars in my bank account",
        "initial_filler": "uh I think I've got about three hundred dollars in my bank account",
        "medial_filler": "I think I've got about uh three hundred dollars in my bank account",
    },
    "heart": {
        "no_filler": "I think it's time you start listening to your heart",
        "initial_filler": "uh I think it's time you start listening to your heart",
        "medial_filler": "I think it's time you uh start listening to your heart",
    },
    "kegs": {
        "no_filler": "I think the kegs are upstairs",
        "initial_filler": "uh I think the kegs are upstairs",
        "medial_filler": "I think the uh kegs are upstairs",
    },
    "misunderstandings": {
        "no_filler": "I think we've had a series of major misunderstandings",
        "initial_filler": "uh I think we've had a series of major misunderstandings",
        "medial_filler": "I think we've had a series of major uh misunderstandings",
    },
    "refund": {
        "no_filler": "I think you should ask for a refund",
        "initial_filler": "um I think you should ask for a refund",
        "medial_filler": "I think you should ask for um a refund",
    },
    "simon": {
        "no_filler": "I think it was simon they were after",
        "initial_filler": "uh I think it was simon they were after",
        "medial_filler": "I think it was uh simon they were after",
    },
    "work": {
        "no_filler": "I think it's gonna work out alright this time",
        "initial_filler": "uh I think it's gonna work out alright this time",
        "medial_filler": "I think it's gonna work out uh alright this time",
    },
}


# find names of all samples
def find_sample_names(dirpath="data/where_is_the_hesitation"):
    wavfiles = glob.glob(join(dirpath, "**/*.wav"), recursive=True)
    names = set()
    for file in wavfiles:
        name = basename(file).split("_")[0].lower()
        names.update([name])
    return list(names)


def save_txt_for_aligner(dirpath=DIRPATH):
    wavfiles = glob.glob(join(dirpath, "**/*.wav"), recursive=True)
    for filepath in wavfiles:
        name = basename(filepath).split("_")[0].lower()
        txtpath = filepath.replace(".wav", ".txt")
        filler_info = basename(dirname(filepath))
        text = name_to_text[name][filler_info]
        write_txt([text], txtpath)


if __name__ == "__main__":
    save_txt_for_aligner()
