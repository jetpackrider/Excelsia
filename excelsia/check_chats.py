"""Check practice chats that were written somewhere else before she trains on them.

    python -m excelsia.check_chats downloaded.jsonl                      # just report
    python -m excelsia.check_chats downloaded.jsonl --into data/chats.jsonl

Every check here is mechanical: a script decides, the same way for every line.
A teacher's own claim that "all lines pass" counts for nothing. The rules are
the ones the built-in chat writer uses:

  - every word of her reply is on her list (plus . , ? ! ; :)
  - her mood is one of the six
  - the reply is a finished sentence, not cut off
  - no word three times running, no sentence said twice in one chat
  - she does not just echo the person, and the person does not repeat themselves

When a line fails, the chat is kept up to the line before it, as long as at
least `--least` exchanges are left. Nothing is ever rewritten here.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .gen_chats import SMALL, sentences
from .tokenizer import MOODS, Tokenizer


def trouble(tok: Tokenizer, turn: object, earlier: list[dict]) -> str | None:
    """Why this exchange cannot be used, or None if it can."""
    if not isinstance(turn, dict) or not all(isinstance(turn.get(k), str) for k in ("user", "mood", "reply")):
        return "not in the right shape"
    user, reply = " ".join(turn["user"].split()), " ".join(turn["reply"].split()).lower()
    words = tok.split(reply)
    if not user or not words:
        return "empty line"
    if len(user) > 300:
        return "the person's message is too long"
    if turn["mood"] not in MOODS:
        return "a mood that is not one of the six"
    ok, bad = tok.can_say(reply)
    if not ok:
        return "a word she cannot say: " + ", ".join(sorted(set(bad))[:4])
    if "".join(words) != "".join(reply.split()):
        return "marks or symbols she cannot say"
    if reply[-1] not in ".?!":
        return "the reply is cut off"
    if any(words[i] == words[i + 1] == words[i + 2] for i in range(len(words) - 2)):
        return "a word three times running"
    if tok.split(user) == words:
        return "she only echoes the person"
    if any(user.lower() == t["user"].lower() for t in earlier):
        return "the person repeats themselves"
    mine = sentences(reply)
    said = {s for t in earlier for s in sentences(t["reply"])}
    if len(set(mine)) < len(mine) or any(s in said for s in mine):
        return "a sentence she already said in this chat"
    return None


def check(tok: Tokenizer, lines: list[str], least: int = 1) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    report = {"read": 0, "whole": 0, "shortened": 0, "dropped": 0, "unreadable": 0,
              "why": collections.Counter(), "words": collections.Counter()}
    for line in lines:
        if not line.strip():
            continue
        report["read"] += 1
        try:
            chat = json.loads(line)
            turns = chat["turns"]
            assert isinstance(turns, list)
        except (ValueError, KeyError, TypeError, AssertionError):
            report["unreadable"] += 1
            continue
        good: list[dict] = []
        for turn in turns:
            why = trouble(tok, turn, good)
            if why:
                report["why"][why.split(":")[0]] += 1
                if why.startswith("a word she cannot say"):
                    report["words"].update(why.split(": ")[1].split(", "))
                break
            good.append({"user": " ".join(turn["user"].split()), "mood": turn["mood"],
                         "reply": " ".join(turn["reply"].split()).lower()})
        if len(good) >= least:
            kept.append({"topic": str(chat.get("topic", ""))[:80], "turns": good})
            report["whole" if len(good) == len(turns) else "shortened"] += 1
        else:
            report["dropped"] += 1
    return kept, report


def describe(tok: Tokenizer, chats: list[dict]) -> str:
    replies = [t["reply"] for c in chats for t in c["turns"]]
    if not replies:
        return "  nothing usable"
    n = len(replies)
    lengths = collections.Counter(len(c["turns"]) for c in chats)
    uses = collections.Counter(w for r in replies for w in set(tok.split(r)) if w.isalpha() and w not in SMALL)
    all_words = {w for r in replies for w in tok.split(r) if w.isalpha()}
    top = ", ".join(f"{w} {100 * k / n:.0f}%" for w, k in uses.most_common(8))
    return (f"  {len(chats)} chats, {n} exchanges, {n / len(chats):.1f} per chat "
            f"({', '.join(f'{k} with {length}' for length, k in sorted(lengths.items()))})\n"
            f"  her replies average {sum(len(r.split()) for r in replies) / n:.1f} words and use "
            f"{len(all_words):,} different words\n"
            f"  asks a question back in {100 * sum('?' in r for r in replies) / n:.0f}% of replies\n"
            f"  most-used words (share of replies): {top}")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m excelsia.check_chats")
    p.add_argument("file", help="a .jsonl file of chats written elsewhere")
    p.add_argument("--into", help="add the chats that pass to this file")
    p.add_argument("--least", type=int, default=1, help="fewest exchanges a chat must keep (default 1)")
    a = p.parse_args()
    tok = Tokenizer()
    kept, r = check(tok, Path(a.file).read_text(encoding="utf-8").splitlines(), a.least)
    print(f"read {r['read']} chats: {r['whole']} pass whole, {r['shortened']} kept but cut short, "
          f"{r['dropped']} dropped, {r['unreadable']} unreadable")
    for why, k in r["why"].most_common():
        print(f"  {k:4d}  {why}")
    if r["words"]:
        print("  words she cannot say that came up most: "
              + ", ".join(f"{w} ({k})" for w, k in r["words"].most_common(15)))
    print(describe(tok, kept))
    if a.into and kept:
        with Path(a.into).open("a", encoding="utf-8") as f:
            for chat in kept:
                f.write(json.dumps(chat, ensure_ascii=False) + "\n")
        print(f"added {len(kept)} chats to {a.into}")


if __name__ == "__main__":
    main()
