"""Build Excelsia's word list from the text of the New York State Constitution.

The rule: Excelsia may only ever say a word that is printed in the body of the
official edition of the constitution. This script turns the raw text extracted
from the official PDF into (1) a cleaned text file and (2) the word list.

    python -m excelsia.words data/source/constitution_raw.txt

Outputs data/ny_constitution.txt and data/excelsia_words.tsv
(word <tab> count <tab> how to print it).
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

RUNNING_HEADER = "The Constitution of the State of New York"
BODY_START = "[Preamble]"

# The PDF's text layer drops the hyphen in a few compound words. Put it back.
FUSED = {
    "attorneygeneral": "attorney-general",
    "lieutenantgovernor": "lieutenant-governor",
    "eightyfour": "eighty-four",
    "fiftyseven": "fifty-seven",
    "sixtythird": "sixty-third",
    "thirtyeight": "thirty-eight",
    "twothirds": "two-thirds",
    "selfpropelled": "self-propelled",
    "stateappropriated": "state-appropriated",
    "codeputy": "co-deputy",
    "coexecutive": "co-executive",
    "colocated": "co-located",
    "omoonshine": "o-moonshine",  # Poke-O-Moonshine Mountain
}

# Things that survive tokenizing but are not words she could sensibly say:
# hyphen prefixes ("re-enacted"), abbreviations ("St. Lawrence", "46kV").
FRAGMENTS = {"re", "bi", "co", "pre", "non", "sub", "st", "kv", "art", "nos"}
# Ordinary words that only happen to appear inside place names ("Lake George").
NOT_NAMES = {"lake", "great", "world", "paper", "freedom", "south", "mountain", "pleasant",
             "sports", "complex", "falls", "forge", "brook", "plains", "minerals", "grid",
             "institute", "tract", "towers", "mount", "moose", "gore", "poke", "gay",
             "indians", "indian", "van", "nordic"}
ROMAN = re.compile(r"^(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def clean_text(raw: str) -> str:
    """Body of the constitution only: no cover page, contents, headers, page numbers."""
    body = raw[raw.index(BODY_START):].replace("\f", "\n")
    lines = []
    for line in body.split("\n"):
        s = line.strip()
        if s == RUNNING_HEADER or re.fullmatch(r"\d{1,3}", s):
            continue
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    seen = collections.Counter(w for w in WORD.findall(text.lower()))

    def rejoin(m: re.Match) -> str:  # "per- mitted" -> "permitted"
        joined = m.group(1) + m.group(2)
        return joined if seen[joined.lower()] else m.group(0)

    text = re.sub(r"\b([A-Za-z]+)- ([a-z]+)\b", rejoin, text)

    def unfuse(m: re.Match) -> str:
        w = m.group(0)
        fixed = FUSED.get(w.lower())
        if not fixed:
            return w
        return fixed.title() if w[0].isupper() and w[1:].islower() else (
            fixed.upper() if w.isupper() else fixed)

    text = re.sub(r"[A-Za-z]+", unfuse, text)
    return text.strip() + "\n"


def is_word(tok: str) -> bool:
    if tok in ("a", "i"):
        return True
    if len(tok) == 1 or tok in FRAGMENTS or ROMAN.match(tok):
        return False
    return True


def count_words(text: str) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for tok in WORD.findall(text.lower()):
        if tok.endswith("'s"):
            tok = tok[:-2]  # "governor's" counts as "governor"
        if is_word(tok):
            counts[tok] += 1
    return counts


def display_forms(text: str, counts: collections.Counter) -> dict[str, str]:
    """Words that are always capitalised in the text (York, Albany, January)."""
    lower: collections.Counter = collections.Counter()
    cap_mid: collections.Counter = collections.Counter()
    for line in text.split("\n"):
        prev = None
        for tok in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^\sA-Za-z]", line):
            if tok[0].isalpha():
                w = tok.lower()
                w = w[:-2] if w.endswith("'s") else w
                if tok[0].islower():
                    lower[w] += 1
                elif len(tok) > 1 and tok[1:].islower() and prev and (prev[0].isalpha() or prev == ","):
                    cap_mid[w] += 1  # capitalised in the middle of a sentence
            prev = tok
    return {w: w.capitalize() for w in counts
            if cap_mid[w] and not lower[w] and w not in NOT_NAMES}


def main(argv: list[str]) -> int:
    raw_path = Path(argv[1]) if len(argv) > 1 else Path("data/source/constitution_raw.txt")
    out_dir = Path("data")
    text = clean_text(raw_path.read_text(encoding="utf-8"))
    (out_dir / "ny_constitution.txt").write_text(text, encoding="utf-8")
    counts = count_words(text)
    shown = display_forms(text, counts)
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    (out_dir / "excelsia_words.tsv").write_text(
        "".join(f"{w}\t{n}\t{shown.get(w, w)}\n" for w, n in rows), encoding="utf-8")
    total = sum(counts.values())
    print(f"{total:,} words in the text, {len(counts):,} unique, "
          f"{sum(1 for n in counts.values() if n > 1):,} used more than once")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
