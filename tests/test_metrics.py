import pytest

from asr_eval.metrics import normalize, ratio_ci, wer_cer, error_counts


def test_normalize():
    assert normalize("  MAYDAY, Mayday!  54°12'N ") == "mayday mayday five four one two n"
    assert normalize("persons-on board: 7.") == "persons on board seven"


def test_identical_texts_have_zero_error():
    assert wer_cer("Silver Heron, 54 north", "silver heron five four north") == (0.0, 0.0)


def test_hand_computed_wer():
    # 6 reference words; "over" added (1 ins), "heron" -> "heroin" (1 sub), "north" dropped (1 del): 3 / 6
    ref, hyp = "this is silver heron five north", "over this is silver heroin five"
    counts = error_counts(ref, hyp)
    assert (counts["sub"], counts["del"], counts["ins"], counts["n_words"]) == (1, 1, 1, 6)
    assert wer_cer(ref, hyp)[0] == pytest.approx(3 / 6)


def test_bootstrap_is_deterministic():
    num, den = [1, 0, 2, 3, 0, 1], [4, 4, 4, 4, 4, 4]
    assert ratio_ci(num, den, seed=0) == ratio_ci(num, den, seed=0)
    low, high = ratio_ci(num, den)
    assert low <= sum(num) / sum(den) <= high
