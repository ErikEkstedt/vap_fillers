from os.path import join, basename
from glob import glob
from tqdm import tqdm
import torch
from vap.utils import read_txt, read_json, write_json

# Copied from website
# http://compprag.christopherpotts.net/swda.html#annotations
DAS = """
   NAME                      act_tag        trainCount   Count    Example     
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

ROOT = "data/swb_da"
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
SPECIAL = [
    "^q",  # Quotation
    "^2",  # Collaborative Completion
    "^h",  # Hold before answer/agreement
    "^g",  # Tag-Question
]

# WARNING: What is '^t' ????????????????
QuestionsInSwb = [
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
    "qy^t",  # Yes-No-Question + WARNING: ????
]


def get_meaning():
    """
    Reads the DAS (above) and organizes it into a dict

    dam = get_meaning()

    print(dam['qy'])
    ----

    {'description': 'Yes-No-Question',
     'da': 'qy',
     'example': 'Do you have to have any special training?'}
    ---
    """
    da_meaning = {}
    for row in DAS.split("\n"):
        if len(row) == 0:
            continue
        _, desc, da, ex, n_train, n_full = row.split("\t")
        n_train = int(n_train)
        n_full = int(n_full)
        da_meaning[da] = {"description": desc, "da": da, "example": ex}
    return da_meaning


def extract_da_data(root, on_da=True):
    """
    Extracts all the dialog-acts, into a dict, ordered by dialog act

    data = extract_da_data(ROOT)

    len(data['qy']) -> 193

    data['qy'][0] ->

        {
            'start': 194.493125,
             'end': 195.953125,
             'words': ['are', 'you', 'talking', 'about', 'like', 'spring', 'break'],
             'starts': [194.493125,
              194.553125,
              194.623125,
              194.793125,
              194.953125,
              195.273125,
              195.623125],
             'ends': [194.553125,
              194.623125,
              194.793125,
              194.953125,
              195.133125,
              195.623125,
              195.953125],
             'boi': ['B', 'I', 'I', 'I', 'I', 'I', 'I'],
             'da': 'qy',
             'da_idx': 'u-42-2',
             'session': '4028',
             'speaker': 'B',
             'text': ' are you talking about like spring break'
        }

    """

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
        da_idx




        """

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
        utt["text"] = text
        return utt

    filepaths = glob(join(root, "**/*.csv"), recursive=True)
    data = {}
    for filepath in tqdm(filepaths):
        name = basename(filepath)
        rows = read_txt(filepath)
        # get_audio_path(name)

        all_rows = read_txt(filepath)
        if len(all_rows) < 1:
            continue

        first_segment = row_to_segment(all_rows[0])
        cur_segments = [first_segment]
        last_da = first_segment["da"]

        for row in all_rows[1:]:
            segment = row_to_segment(row)
            # utt_id, start, end, word, boi, da, da_idx = row.split(",")

            if on_da:
                if segment["da"] != last_da:
                    if last_da not in data:
                        data[last_da] = []

                    # Add last segments
                    utt = segments_to_utt(cur_segments)
                    data[last_da].append(utt)

                    # start new segments
                    cur_segments = [segment]
                    last_da = segment["da"]
                else:
                    cur_segments.append(segment)
            else:
                if segment["boi"] == "B":
                    if last_da not in data:
                        data[last_da] = []
                    # add last_current
                    utt = segments_to_utt(cur_segments)
                    data[last_da].append(utt)
                    # new current semgent
                    cur_segments = [segment]
                    last_da = segment["da"]
                else:
                    cur_segments.append(segment)

        # Add last
        utt = segments_to_utt(cur_segments)
        if last_da not in data:
            data[last_da] = []
        data[last_da].append(utt)
    return data


if __name__ == "__main__":

    dam = get_meaning()
    data = extract_da_data(ROOT)
    # data = extract_da_data(root, on_da=False)
    # write_json(data, 'data/da_utterances.json')

    data = read_json("data/da_utterances.json")

    for da, desc in dam.items():
        if da.startswith("^"):
            print(f"{da}: {desc['description']}")

    # qy^d   -> Declarative Yes-No-Question
    # ^h     -> Hold before answer/agreement
    # qy^d^h -> Declarative YNQ and  hold before agreement
    print("QUESTIONS")
    for q in QUESTIONS:
        print(f"{q}: {dam[q]['description']}")

    print("BACKCHANNEL")
    for b in BACKCHANNELS:
        print(f"{b}: {dam[b]['description']}")

    for da in data.keys():
        if da.startswith("q"):
            print(da)

    print(dam["^g"])

    # data_values = []
    # data_keys = []
    for k, v in data.items():
        print(f"{k}: {len(v)}")
        # data_keys.append(k)
        # data_values.append(v)

    n_words_bc = torch.zeros(5)
    for bc in data["b"]:
        n_words_bc[len(bc) - 1] += 1
        if len(bc) == 3:
            for b in bc:
                print(b)
            input()

    for k, v in dam.items():
        if k.startswith("q"):
            print(f'{k}: {v["description"]}')
