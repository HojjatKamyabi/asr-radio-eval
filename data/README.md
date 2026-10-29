# Data

`messages.csv` holds 30 fictional MAYDAY messages from `scripts/make_messages.py` (seed 42). The vessel names, positions, distress types and crew counts are all made up.

`synthetic/<id>.flac` contains the messages spoken by Piper TTS, resampled from 22.05 kHz to 16 kHz mono, 16-bit FLAC. The files are 15.6 to 18.2 s long, 9.8 MB in total.

| Voice | Speaker | Messages | Training data | Licence |
|---|---|---|---|---|
| `en_US-ljspeech-medium` | US English, female | ids 1, 4, 7, ... | LJ Speech | public domain |
| `en_US-norman-medium` | US English, male | ids 2, 5, 8, ... | LibriVox recordings | public domain |
| `en_GB-cori-medium` | British English, female | ids 3, 6, 9, ... | LibriVox recordings | public domain |

The licences come from each voice's `MODEL_CARD` in [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices). Synthesis used `piper-tts==1.8.0` (GPL-3.0), called as an external program by `scripts/synthesize.py`. None of this repository's code imports Piper.

To regenerate the audio:

```bash
pip install -r requirements-data.txt
python -m piper.download_voices --download-dir voices en_US-ljspeech-medium en_US-norman-medium en_GB-cori-medium
python scripts/synthesize.py
```

Piper adds random noise when it generates speech, so regenerated audio may not be bit-identical to the committed files. The committed files are the ones used for the results.

`real/` is for optional recordings of the same 30 texts (`<id>.wav`). If the folder exists, `run_eval.py` evaluates it as well.
