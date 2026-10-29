# Transcribe every message in every channel condition and score WER and facts.
# Usage: python scripts/run_eval.py --model tiny.en
# Results go to results/<model>/synthetic/ (and results/<model>/real/ if data/real/ exists).
import argparse
import csv
import datetime
import json
import platform
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asr_eval.degrade import CONDITIONS, degrade, noise_only
from asr_eval.extract import extract, truth_from_row
from asr_eval.metrics import SLOTS, error_counts, normalize, ratio_ci, score_facts
from asr_eval.transcribe import ASR_SETTINGS, DEFAULT_MODEL, load_audio, load_model, transcribe

DATASETS = {"synthetic": "flac", "real": "wav"}
PACKAGES = ["faster-whisper", "ctranslate2", "jiwer", "numpy", "scipy", "soundfile", "matplotlib"]


def to_text(value):
    # write facts to the CSV: None -> "", (53, 35, 6, 1) -> "53 35 6 1"
    if value is None:
        return ""
    if isinstance(value, tuple):
        return " ".join(str(v) for v in value)
    return value


def evaluate_clip(model, row, audio, condition):
    hyp = transcribe(model, degrade(audio, condition, seed=int(row["id"])))
    c = error_counts(row["reference"], hyp)
    facts = extract(hyp)
    outcomes = score_facts(facts, truth_from_row(row))
    clip = {
        "id": row["id"],
        "condition": condition,
        "reference": row["reference"],
        "hypothesis": hyp,
        "wer": round((c["sub"] + c["del"] + c["ins"]) / c["n_words"], 4),
        "cer": round(c["char_errors"] / c["n_chars"], 4),
        "sub": c["sub"],
        "del": c["del"],
        "ins": c["ins"],
    }
    for slot in SLOTS:
        clip[slot] = to_text(facts[slot])
        clip[slot + "_outcome"] = outcomes[slot]
    return clip, c


def summarize(clips, counts):
    rows = []
    for condition in CONDITIONS:
        idx = [i for i, clip in enumerate(clips) if clip["condition"] == condition]
        word_errors = [counts[i]["sub"] + counts[i]["del"] + counts[i]["ins"] for i in idx]
        n_words = [counts[i]["n_words"] for i in idx]
        wrong = [sum(clips[i][s + "_outcome"] == "wrong" for s in SLOTS) for i in idx]
        missing = [sum(clips[i][s + "_outcome"] == "missing" for s in SLOTS) for i in idx]
        n_facts = len(idx) * len(SLOTS)
        row = {
            "condition": condition,
            "n_messages": len(idx),
            "wer": sum(word_errors) / sum(n_words),
            "cer": sum(counts[i]["char_errors"] for i in idx) / sum(counts[i]["n_chars"] for i in idx),
            "wrong_fact_rate": sum(wrong) / n_facts,
            "missing_fact_rate": sum(missing) / n_facts,
        }
        row["wer_ci_low"], row["wer_ci_high"] = ratio_ci(word_errors, n_words)
        row["wrong_ci_low"], row["wrong_ci_high"] = ratio_ci(wrong, [len(SLOTS)] * len(idx))
        for slot in SLOTS:
            for result in ["correct", "missing", "wrong"]:
                row[slot + "_" + result] = sum(clips[i][slot + "_outcome"] == result for i in idx) / len(idx)
        rows.append({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    return rows


def plot(summary, path, title):
    x = range(len(summary))
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, ci, label in [("wer", "wer", "WER"), ("wrong_fact_rate", "wrong", "Wrong-fact rate")]:
        y = [r[key] for r in summary]
        low = [r[key] - r[ci + "_ci_low"] for r in summary]
        high = [r[ci + "_ci_high"] - r[key] for r in summary]
        ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=4, label=label)
    ax.plot(x, [r["missing_fact_rate"] for r in summary], "o--", color="grey", label="Missing-fact rate")
    ax.set_xticks(list(x), [r["condition"] for r in summary])
    ax.set_ylabel("rate (95% bootstrap CI)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(model, model_name, dataset, rows, out):
    out.mkdir(parents=True, exist_ok=True)
    clips, counts, noise = [], [], []
    for row in rows:
        audio = load_audio(str(ROOT / "data" / dataset / f"{row['id']}.{DATASETS[dataset]}"))
        for condition in CONDITIONS:
            clip, c = evaluate_clip(model, row, audio, condition)
            clips.append(clip)
            counts.append(c)
        hyp = transcribe(model, noise_only(audio, seed=int(row["id"])))
        noise.append({"id": row["id"], "hypothesis": hyp, "normalized": normalize(hyp)})
        print(dataset, row["id"], "done", flush=True)

    summary = summarize(clips, counts)
    hallucinated = [n["normalized"] for n in noise if n["normalized"]]
    summary.append({
        "condition": "noise_only",
        "n_messages": len(noise),
        "hallucination_rate": round(len(hallucinated) / len(noise), 4),
    })
    columns = list(summary[0]) + ["hallucination_rate"]
    write_csv(out / "per_clip.csv", clips)
    write_csv(out / "summary.csv", [{k: r.get(k, "") for k in columns} for r in summary])
    write_csv(out / "hallucinations.csv", noise)
    plot(summary[:-1], out / "figure.png", f"{model_name}, {dataset} speech (n = {len(rows)} messages)")
    print("most common hallucinations:", Counter(hallucinated).most_common(5))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    messages = list(csv.DictReader(open(ROOT / "data" / "messages.csv", encoding="utf-8")))
    model = load_model(args.model)

    results = {}
    for dataset, ext in DATASETS.items():
        rows = [r for r in messages if (ROOT / "data" / dataset / f"{r['id']}.{ext}").exists()]
        if not rows:
            continue
        out = ROOT / "results" / args.model / dataset
        results[dataset] = run_dataset(model, args.model, dataset, rows, out)
        info = {
            "model": args.model,
            "dataset": dataset,
            "n_messages": len(rows),
            "device": "cpu",
            "compute_type": "int8",
            "asr_settings": ASR_SETTINGS,
            "seeds": {"messages": 42, "noise": "clip id", "bootstrap": 0, "bootstrap_resamples": 1000},
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {p: version(p) for p in PACKAGES},
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        (out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")

    # synthetic and real side by side, only if real recordings exist
    if "real" in results:
        rows = []
        for synthetic, real in zip(results["synthetic"][:-1], results["real"][:-1]):
            row = {"condition": synthetic["condition"]}
            for key in ["wer", "wrong_fact_rate", "missing_fact_rate"]:
                row[key + "_synthetic"] = synthetic[key]
                row[key + "_real"] = real[key]
            rows.append(row)
        write_csv(ROOT / "results" / args.model / "comparison.csv", rows)


if __name__ == "__main__":
    main()
