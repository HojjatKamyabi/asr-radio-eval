# Text normalization, WER/CER, fact scoring and bootstrap confidence intervals.
import re

import jiwer
import numpy as np

DIGIT_WORDS = "zero one two three four five six seven eight nine".split()
SLOTS = ["vessel", "position", "distress", "persons"]


def normalize(text):
    # lowercase, remove punctuation, "54" -> "five four"
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    text = re.sub(r"\d", lambda m: " " + DIGIT_WORDS[int(m.group())] + " ", text)
    return " ".join(text.split())


def error_counts(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    words = jiwer.process_words(ref, hyp)
    chars = jiwer.process_characters(ref, hyp)
    return {
        "sub": words.substitutions,
        "del": words.deletions,
        "ins": words.insertions,
        "n_words": words.substitutions + words.deletions + words.hits,
        "char_errors": chars.substitutions + chars.deletions + chars.insertions,
        "n_chars": chars.substitutions + chars.deletions + chars.hits,
    }


def wer_cer(reference, hypothesis):
    c = error_counts(reference, hypothesis)
    wer = (c["sub"] + c["del"] + c["ins"]) / c["n_words"]
    cer = c["char_errors"] / c["n_chars"]
    return wer, cer


def score_facts(extracted, truth):
    result = {}
    for slot in SLOTS:
        if extracted[slot] is None:
            result[slot] = "missing"
        elif extracted[slot] == truth[slot]:
            result[slot] = "correct"
        else:
            result[slot] = "wrong"
    return result


def ratio_ci(numerators, denominators, n_resamples=1000, seed=0):
    # 95% bootstrap CI of sum(numerators) / sum(denominators), resampling messages
    num = np.asarray(numerators, float)
    den = np.asarray(denominators, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_resamples, len(num)))
    stats = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
