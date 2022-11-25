# Hesitation


Samples from [Where's the uh, hesitation? The interplay between filled pause location, speech rate and fundamental frequency](https://www.speech.kth.se/tts-demos/interspeech2022/)



* Create txt-files for all wav-files
```bash
python prepare_forced_aligner.py
```
* run forced aligner. Montreal forced aligner "should" be installed in its own conda environment. then run:
```bash
conda run -n aligner mfa align data/where_is_the_hesitation/Stimuli english_us_arpa english_us_arpa data/where_is_the_hesitation/alignment
```
* Given the previous script TextGrid-files are created in `data/where_is_the_hesitation/alignment` with the same directory structure as the original data.

