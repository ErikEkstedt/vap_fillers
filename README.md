# Role of Fillers: Analyzes with Voice Activity Projection models


What is the role of fillers ('eh', 'um') according to Voice Activity Projection objective?

* Does it consider a filler as a turn-holding cue?
* What is the difference in turn-holding strength with/without a filler?


```bash
.
├── data
│   ├── relative_audio_path.json
│   └── test.txt
├── hesitation
│   └── wheres_the_hesitation.py
├── filler_effect.py
├── LICENSE
└── README.md
```


* `data/`
    - `test.txt`: sessions in switchboard DAMSL split not used during training
    - `relative_audio_path.json`: helper mapping to map sessions (switchboard) to their relative audio path

* `filler_effect.py`
    - script to extract next speaker probabilities
