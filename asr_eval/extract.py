# Find the four facts (vessel, position, distress, persons) in a transcript with simple rules.
import re
from collections import Counter

from asr_eval.metrics import DIGIT_WORDS, normalize

DIGITS = {word: str(i) for i, word in enumerate(DIGIT_WORDS)}
DISTRESS_PATTERNS = {
    "fire": r"\bfire\b",
    "flooding": r"\bflood(ing|ed)?\b",
    "collision": r"\bcollision\b|\bcollided\b",
    "grounding": r"\bground(ing|ed)\b|\baground\b",
    "man overboard": r"\bover ?board\b",
}
PERSON_WORDS = {"persons", "person", "people"}
POSITION_END = {"east", "e", "nature", "distress"} | PERSON_WORDS


def extract_vessel(tokens):
    # take the words between "this is" and "my"/"position", then the most common word pair
    for i in range(len(tokens) - 1):
        if tokens[i] == "this" and tokens[i + 1] == "is":
            words = []
            for tok in tokens[i + 2 :]:
                if tok in ("my", "position"):
                    break
                words.append(tok)
            pairs = [words[j] + " " + words[j + 1] for j in range(len(words) - 1)]
            if not pairs:
                return None
            return Counter(pairs).most_common(1)[0][0]
    return None


def extract_position(tokens):
    if "position" not in tokens:
        return None
    # collect groups of digits after "position", e.g. ["53", "35", "006", "01"]
    groups, current = [], ""
    for tok in tokens[tokens.index("position") + 1 :]:
        if tok in POSITION_END:
            break
        if tok in DIGITS:
            current += DIGITS[tok]
        elif current:
            groups.append(current)
            current = ""
    if current:
        groups.append(current)

    if len(groups) == 4:
        position = tuple(int(g) for g in groups)
    else:
        # digits without words in between: latitude is always 4 digits, the rest is longitude
        digits = "".join(groups)
        if len(digits) < 7:
            return None
        position = (int(digits[0:2]), int(digits[2:4]), int(digits[4:-2]), int(digits[-2:]))

    # an impossible position would be noticed by the operator, so count it as missing
    lat_deg, lat_min, lon_deg, lon_min = position
    if lat_deg > 90 or lon_deg > 180 or lat_min >= 60 or lon_min >= 60:
        return None
    return position


def extract_distress(text):
    # the distress keyword that appears first in the text
    found = []
    for name, pattern in DISTRESS_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            found.append((match.start(), name))
    if not found:
        return None
    return min(found)[1]


def extract_persons(tokens):
    # the digits right before "persons"
    for i, tok in enumerate(tokens):
        if tok in PERSON_WORDS:
            digits = ""
            for prev in reversed(tokens[:i]):
                if prev not in DIGITS:
                    break
                digits = DIGITS[prev] + digits
            if digits:
                return int(digits)
    return None


def extract(text):
    text = normalize(text)
    tokens = text.split()
    return {
        "vessel": extract_vessel(tokens),
        "position": extract_position(tokens),
        "distress": extract_distress(text),
        "persons": extract_persons(tokens),
    }


def truth_from_row(row):
    # true facts from a messages.csv row, in the same format as extract()
    return {
        "vessel": row["vessel"].lower(),
        "position": tuple(int(x) for x in row["position"].split()),
        "distress": row["distress"],
        "persons": int(row["persons"]),
    }
