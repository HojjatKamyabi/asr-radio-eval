# Turn data/messages.csv into speech with Piper -> data/synthetic/<id>.flac (16 kHz mono).
# Piper is run as a separate program (not imported). See data/README.md for setup.
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
VOICES = ROOT / "voices"
OUT = ROOT / "data" / "synthetic"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "data" / "messages.csv", encoding="utf-8")))
    with tempfile.TemporaryDirectory() as tmp:
        for row in rows:
            wav = Path(tmp) / f"{row['id']}.wav"
            voice = VOICES / f"{row['voice']}.onnx"
            cmd = [sys.executable, "-m", "piper", "-m", str(voice), "-f", str(wav)]
            subprocess.run(cmd, input=row["reference"], text=True, check=True)
            audio, rate = sf.read(wav)
            audio = resample_poly(audio, 320, 441)  # Piper gives 22050 Hz, we want 16000 Hz
            sf.write(OUT / f"{row['id']}.flac", audio.clip(-1, 1), 16000, subtype="PCM_16")
            print(row["id"], row["voice"], f"{len(audio) / 16000:.1f} s")


if __name__ == "__main__":
    main()
