# One real test with the tiny.en model (downloads about 75 MB the first time).
import csv
from pathlib import Path

import pytest

from asr_eval.metrics import wer_cer
from asr_eval.transcribe import load_audio, load_model, transcribe

DATA = Path(__file__).parent.parent / "data"


@pytest.mark.slow
def test_tiny_en_on_clean_clip():
    reference = next(csv.DictReader(open(DATA / "messages.csv", encoding="utf-8")))["reference"]
    text = transcribe(load_model("tiny.en"), load_audio(str(DATA / "synthetic" / "01.flac")))
    # WER was 0.079 on Windows and 0.105 on Linux, so 0.25 leaves a clear margin
    assert wer_cer(reference, text)[0] < 0.25
