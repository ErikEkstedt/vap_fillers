from os.path import join, basename
from pathlib import Path
from glob import glob
from tqdm import tqdm
import pandas as pd
import torch
from vap.utils import read_txt, read_json, write_json

SWDA_WORDS = "data/swb_dialog_acts_words"
SWDA_UTT = "data/swb_dialog_acts_utterances"
NAMES = ["utt_idx", "start", "end", "word", "boi", "da", "da_idx"]
SPECIAL = [
    "^q",  # Quotation
    "^2",  # Collaborative Completion
    "^h",  # Hold before answer/agreement
    "^g",  # Tag-Question
]
QUESTIONS = [
    "qy",
    "qw",
    "qy^d",
    "qo",
    "qh",
    "qrr",
    "qw^d",
]
BACKCHANNELS = ["b", "bh"]
Q_IN_SWB = [
    "qh",  # Rhetorical-Questions
    "qo",  #  Open-Question
    "qr",  #  Or-Clause ? seems to be? WARNING:
    "qr^d",  # Declarative Or-Clause ? seems to be? WARNING:
    "qrr",  # Or-Clause
    "qw",  # Wh-Question
    "qw^d",  # Wh-Question Declarative
    "qw^h",  # Wh-Question + "Hold before answer/agreement"
    "qy",  # Yes-No-Question
    "qy^d",  # Declarative-Question
    "qy^d^h",  # Declarative-Question + "Hold before answer/agreement"
    "qy^g",  # Yes-No-Question + "Tag-question = 'Right?'"
    "qy^g^t",  # Yes-No-Question + "Tag-question = 'Right?'" + WARNING:
    "qy^h",  # Yes-No-Question + "Hold before answer/agreement"
    "qy^m",  # Yes-No-Question + WARNING: ?????????
    "qy^r",  # Yes-No-Question + WARNING: ????
    "qy^t",  # Yes-No-Question + WARNING: What is '^t' ????????????????
]


def get_description():
    """
    Copied from website
        http://compprag.christopherpotts.net/swda.html#annotations

    idx     NAME    act_tag     trainCount  Count   Example
    """

    das = """
    1  Statement-non-opinion        sd            72824      75145    "Me, I'm in the legal department."                  
    2  Acknowledge-(Backchannel)    b             37096      38298    "Uh-huh. "                                          
    3  Statement-opinion            sv            25197      26428    "I think it's great "                               
    4  Agree/Accept                 aa            10820      11133    "That's exactly it. "                               
    5  Abandoned-or-Turn-Exit       %             10569      15550    "So, - "                                            
    6  Appreciation                 ba            4633       4765     "I can imagine. "                                   
    7  Yes-No-Question              qy            4624       4727     "Do you have to have any special training? "        
    8  Non-verbal                   x             3548       3630     "[Laughter], [Throat_clearing] "                    
    9  Yes-answers                  ny            2934       3034     "Yes. "                                             
    10 Conventional-closing         fc            2486       2582     "Well, it's been nice talking to you. "             
    11 Uninterpretable              %             2158       15550    "But, uh, yeah "                                    
    12 Wh-Question                  qw            1911       1979     "Well, how old are you? "                           
    13 No-answers                   nn            1340       1377     "No. "                                              
    14 Response-Acknowledgement     bk            1277       1306     "Oh, okay. "                                        
    15 Hedge                        h             1182       1226     "I don't know if I'm making any sense or not. "     
    16 Declarative-Yes-No-Question  qy^d          1174       1219     "So you can afford to get a house? "                
    17 Other                        fo_o_fw_by_bc 1074       883      "Well give me a break, you know. "                  
    18 Backchannel-in-question-form bh            1019       1053     "Is that right? "                                   
    19 Quotation                    ^q            934        983      "You can't be pregnant and have cats "              
    20 Summarize/reformulate        bf            919        952      "Oh, you mean you switched schools for the kids. "  
    21 Affirmative-non-yes-answers  na            836        847      "It is. "                                           
    22 Action-directive             ad            719        746      "Why don't you go first "                           
    23 Collaborative-Completion     ^2            699        723      "Who aren't contributing. "                         
    24 Repeat-phrase                b^m           660        688      "Oh, fajitas "                                      
    25 Open-Question                qo            632        656      "How about you? "                                   
    26 Rhetorical-Questions         qh            557        575      "Who would steal a newspaper? "                     
    27 Hold-before-answer/agreement ^h            540        556      "I'm drawing a blank. "                             
    28 Reject                       ar            338        346      "Well, no "                                         
    29 Negative-non-no-answers      ng            292        302      "Uh, not a whole lot. "                             
    30 Signal-non-understanding     br            288        298      "Excuse me? "                                       
    31 Other-answers                no            279        286      "I don't know "                                     
    32 Conventional-opening         fp            220        225      "How are you? "                                     
    33 Or-Clause                    qrr           207        209      "or is it more of a company? "                      
    34 Dispreferred-answers         arp_nd        205        207      "Well, not so much that. "                          
    35 3rd-party-talk               t3            115        117      "My goodness, Diane, get down from there. "         
    36 Offers,Options,Commits       oo_co_cc      109        110      "I'll have to check that out "                      
    37 Self-talk                    t1            102        103      "What's the word I'm looking for "                  
    38 Downplayer                   bd            100        103      "That's all right. "                                
    39 Maybe/Accept-part            aap_am        98         105      "Something like that "                              
    40 Tag-Question                 ^g            93         92       "Right? "                                           
    41 Declarative-Wh-Question      qw^d          80         80       "You are what kind of buff? "                       
    42 Apology                      fa            76         79       "I'm sorry. "                                       
    43 Thanking                     ft            67         78       "Hey thanks a lot "                                 
    """

    description = {}
    for row in das.split("\n"):
        # print(len(row))
        row = row.strip()
        if len(row) == 0:
            continue
        idx, desc, da, n_train, n_full, *ex = row.split()
        n_train = int(n_train)
        n_full = int(n_full)
        description[da] = {"description": desc, "da": da, "example": ex}

    return description


def save_utterance_da(word_root, savedir="data/swb_dialog_acts_utterance"):
    Path(savedir).mkdir(parents=True, exist_ok=True)
    for filepath in tqdm(
        glob(join(word_root, "*.csv")), desc="Extract Dialog Act Utterances"
    ):
        w = pd.read_csv(filepath, names=NAMES)
        utts = []
        for utt_idx in w["utt_idx"].unique():
            utt = w[w["utt_idx"] == utt_idx]

            # There is obviously a better way of doing this
            # but I suck at pandas...
            # only performed 'once'... in theory...
            utts.append(
                {
                    "utt_idx": utt_idx,
                    "start": utt.start.values.tolist()[0],
                    "end": utt.start.values.tolist()[-1],
                    "da": utt.da.values.tolist(),
                    "da_unique": utt.da.unique().tolist(),
                    "starts": utt.start.values.tolist(),
                    "ends": utt.end.values.tolist(),
                    "words": utt.word.values.tolist(),
                    "text": " ".join(utt.word.values.tolist()),
                }
            )
        # df = pd.DataFrame(utts)
        # filename = basename(filepath).replace("word", "utt")
        # savepath = join(savedir, filename)
        # df.to_csv(savepath, index=False)
        filename = basename(filepath).replace("word", "utt").replace(".csv", ".json")
        savepath = join(savedir, filename)
        write_json(utts, savepath)


if __name__ == "__main__":
    dad = get_description()

    save_utterance_da(SWDA_WORDS, SWDA_UTT)
