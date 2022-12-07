# Role of Fillers: Analyzes with Voice Activity Projection models


What is the role of fillers ('eh', 'um') according to Voice Activity Projection objective?

* Does it consider a filler as a turn-holding cue?


## Rund

1. Extract Fillers: 
    - `python vap_fillers/filler_extraction_from_transcript.py`
2. Extract Global Prosody: 
    - `python vap_fillers/filler_prosody.py --aggregate` 
3. Extract utterance and filler level prosody
    - `python vap_fillers/filler_prosody.py --filler` 
4. Extract model output
    - `python vap_fillers/filler_model_process.py` 
5. Visualize fillers (streamlit)
    - `python vap_fillers/filler_visualization.py`


--------------------------------------------

```
USER:
----- 
hello there, I am investigating the role of 'fillers' in spoken dialog. Could
you give me a brief overview of what they are and some interesting ideas to
investigating regarding their role in speech?

ChatGPT:
--------
Hello! Fillers are words or sounds that are used to fill gaps in speech, such
as "um" or "uh." They can serve a variety of purposes, such as signaling to the
listener that the speaker is still thinking or organizing their thoughts, or as
a way of checking in with the listener to make sure they are still paying
attention. Investigating the role of fillers in spoken dialog could involve
looking at factors such as how often they are used, in what contexts they are
used, and how they are perceived by listeners. It could also involve looking at
whether there are any differences in the use of fillers between different
speakers or in different speaking situations. Overall, there are many
interesting directions that research on fillers in spoken dialog could take,
and I'm sure you'll be able to come up with some interesting ideas for your
investigation!


USER:
-----
thanks, do you know if there  is a difference between 'uh' and 'um'?

ChatGPT:
--------
Yes, there is a difference between "uh" and "um." In general, "uh" is used to
signal a pause in speech to indicate that the speaker is unsure or uncertain
about what they are about to say. "Um," on the other hand, is used to signal a
pause in speech to indicate that the speaker is thinking or organizing their
thoughts. Some studies have also suggested that "uh" is used more often by men,
while "um" is used more often by women, although this is not always the case.
Overall, the specific meaning and use of "uh" and "um" can vary depending on
the speaker and the context in which they are used.
```
