import csv
from pathlib import Path

import pytest

from asr_eval.extract import extract, truth_from_row
from asr_eval.metrics import score_facts

ROWS = list(csv.DictReader(open(Path(__file__).parent.parent / "data" / "messages.csv", encoding="utf-8")))


def test_thirty_messages():
    assert len(ROWS) == 30


@pytest.mark.parametrize("row", ROWS, ids=[r["id"] for r in ROWS])
def test_extractor_is_perfect_on_references(row):
    assert extract(row["reference"]) == truth_from_row(row)


TRUTH = {"vessel": "amber gannet", "position": (53, 35, 6, 1), "distress": "flooding", "persons": 24}


def test_whisper_style_digits_are_understood():
    text = "Mayday. This is Amber Gannet, Amber Gannet. My position 53 degrees 35 minutes north, 006 degrees 01 minutes east. Flooding. 24 persons on board. Over."
    assert score_facts(extract(text), TRUTH) == dict.fromkeys(TRUTH, "correct")


def test_missing_slots():
    text = "mayday mayday this amber gannet my position five three degrees north nature of distress over"
    assert score_facts(extract(text), TRUTH) == dict.fromkeys(TRUTH, "missing")


def test_impossible_position_is_missing():
    text = "my position five three degrees three five minutes north seven zero zero seven degrees five five minutes east"
    assert extract(text)["position"] is None


def test_wrong_slots():
    text = ("this is amber garnet amber garnet amber gannet my position five three degrees three nine minutes north "
            "zero zero six degrees zero one minutes east nature of distress fire two four persons on board over")
    result = score_facts(extract(text), TRUTH)
    assert result == {"vessel": "wrong", "position": "wrong", "distress": "wrong", "persons": "correct"}
