"""Word-level tokenizer for Excelsia.

Two kinds of words live in one vocabulary:

  * speaking words  - every word printed in the New York State Constitution.
                      These are the ONLY words she can say.
  * reading words   - ordinary English she can understand but never say.
                      Her own name is one of these: she knows it when she
                      hears it, but "Excelsia" is not in the constitution.

`Tokenizer.speak_mask` marks the tokens she is allowed to produce. Generation
sets every other token's score to minus infinity, so a word outside the
constitution is not unlikely, it is impossible.

    python -m excelsia.tokenizer          # rebuild data/vocab.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

MOODS = ["grateful", "pleasure", "peace", "alone", "pain", "offense"]
SPECIALS = ["<pad>", "<unk>", "<story>", "<user>", "<excelsia>", "<end>"] + [
    f"<mood:{m}>" for m in MOODS]
PUNCT = [".", ",", "?", "!", ";", ":", "'", '"', "-", "(", ")"]
SPEAK_PUNCT = [".", ",", "?", "!", ";", ":"]
NAME = "excelsia"

_TOKEN = re.compile(r"<[a-z:]+>|[a-z]+(?:'[a-z]+)?|[0-9]+|[.,?!;:'\"()\-]")
_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                       "—": "-", "–": "-"})


def build_vocab() -> dict:
    """Assemble the vocabulary from the two word lists in data/."""
    speak, display = [], {}
    for line in (DATA / "excelsia_words.tsv").read_text(encoding="utf-8").splitlines():
        word, _count, shown = line.split("\t")
        speak.append(word)
        if shown != word:
            display[word] = shown
    speak = sorted(set(speak))
    display["i"] = "I"
    read_only = (DATA / "input_words.txt").read_text(encoding="utf-8").split()
    read_only = sorted((set(read_only) | {NAME}) - set(speak))
    return {"specials": SPECIALS, "punct": PUNCT, "speak": speak,
            "read_only": read_only, "display": display}


class Tokenizer:
    def __init__(self, vocab: dict | None = None):
        if vocab is None:
            vocab = json.loads((DATA / "vocab.json").read_text(encoding="utf-8"))
        self.vocab = vocab
        self.itos = vocab["specials"] + vocab["punct"] + vocab["speak"] + vocab["read_only"]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        assert len(self.stoi) == len(self.itos), "duplicate tokens in vocabulary"
        self.display = vocab["display"]
        self.pad, self.unk = self.stoi["<pad>"], self.stoi["<unk>"]
        self.story, self.user = self.stoi["<story>"], self.stoi["<user>"]
        self.excelsia, self.end = self.stoi["<excelsia>"], self.stoi["<end>"]
        self.speak_words = set(vocab["speak"])
        allowed = self.speak_words | set(SPEAK_PUNCT) | {"<end>"}
        self.speak_mask = [tok in allowed for tok in self.itos]

    def __len__(self) -> int:
        return len(self.itos)

    def mood(self, name: str) -> int:
        return self.stoi[f"<mood:{name}>"]

    def split(self, text: str) -> list[str]:
        return _TOKEN.findall(text.lower().translate(_FOLD))

    def encode(self, text: str) -> list[int]:
        """Text to token ids. Words she has never seen become <unk>."""
        out = []
        for tok in self.split(text):
            idx = self.stoi.get(tok)
            if idx is None and tok.endswith("'s"):  # "york's" reads as "york"
                idx = self.stoi.get(tok[:-2])
            out.append(self.unk if idx is None else idx)
        return out

    def can_say(self, text: str) -> tuple[bool, list[str]]:
        """Is every word in `text` one she is allowed to say? Returns the offenders."""
        allowed = self.speak_words | set(SPEAK_PUNCT)
        bad = [t for t in self.split(text) if t not in allowed]
        return (not bad, bad)

    def decode(self, ids: list[int]) -> str:
        """Token ids back to readable text, with capitals and spacing restored."""
        out, start = [], True
        for i in ids:
            tok = self.itos[i]
            if tok in ("<pad>", "<end>", "<excelsia>") or tok.startswith("<mood:"):
                continue
            if tok in (".", ",", "?", "!", ";", ":", ")"):
                out.append(tok)
                start = start or tok in ".?!"
                continue
            word = self.display.get(tok, tok)
            if start and word[:1].isalpha():
                word = word[0].upper() + word[1:]
            out.append((" " if out and out[-1] != "(" else "") + word)
            start = False
        text = "".join(out).strip()
        return re.sub(r"\b[Nn]ew York\b", "New York", text)


def main() -> None:
    vocab = build_vocab()
    (DATA / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    tok = Tokenizer(vocab)
    print(f"{len(tok):,} tokens: {len(vocab['speak']):,} she can say, "
          f"{len(vocab['read_only']):,} she can only read")


if __name__ == "__main__":
    main()
