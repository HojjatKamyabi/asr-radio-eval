# Make 30 fictional MAYDAY messages with known facts -> data/messages.csv
import csv
import random
from pathlib import Path

SEED = 42
N_MESSAGES = 30
VOICES = ["en_US-ljspeech-medium", "en_US-norman-medium", "en_GB-cori-medium"]
FIRST = ["Silver", "Amber", "Quiet", "Golden", "Morning", "Copper"]
SECOND = ["Heron", "Gannet", "Lantern", "Meadow", "Harvest", "Willow"]
DISTRESS = ["fire", "flooding", "collision", "grounding", "man overboard"]
DIGITS = "zero one two three four five six seven eight nine".split()
OUT = Path(__file__).resolve().parent.parent / "data" / "messages.csv"


def spell(number, width=1):
    # spell a number digit by digit: spell(7, 3) -> "zero zero seven"
    return " ".join(DIGITS[int(d)] for d in str(number).zfill(width))


def make_message(rng, vessel):
    lat = rng.randint(53 * 60, 55 * 60)  # in minutes, between 53°N and 55°N
    lon = rng.randint(6 * 60, 9 * 60)  # in minutes, between 6°E and 9°E
    lat_deg, lat_min, lon_deg, lon_min = lat // 60, lat % 60, lon // 60, lon % 60
    distress = rng.choice(DISTRESS)
    persons = rng.randint(2, 25)
    text = (
        f"Mayday, mayday, mayday. This is {vessel}, {vessel}, {vessel}. "
        f"My position {spell(lat_deg, 2)} degrees {spell(lat_min, 2)} minutes north, "
        f"{spell(lon_deg, 3)} degrees {spell(lon_min, 2)} minutes east. "
        f"Nature of distress, {distress}. "
        f"{spell(persons).capitalize()} persons on board. Over."
    )
    return {
        "reference": text,
        "vessel": vessel,
        "position": f"{lat_deg} {lat_min} {lon_deg} {lon_min}",
        "distress": distress,
        "persons": persons,
    }


def main():
    rng = random.Random(SEED)
    names = rng.sample([f"{a} {b}" for a in FIRST for b in SECOND], N_MESSAGES)
    rows = []
    for i, name in enumerate(names):
        row = {"id": f"{i + 1:02d}", "voice": VOICES[i % len(VOICES)]}
        row.update(make_message(rng, name))
        rows.append(row)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} messages to {OUT}")


if __name__ == "__main__":
    main()
