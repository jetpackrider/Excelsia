"""The part of the Discord bot that has nothing to do with Discord: who is talking to
her where, and what she says back. Kept separate so it can be tested on its own."""
from __future__ import annotations

import re
import time

from excelsia.lite import Excelsia, LiteGPT
from excelsia.tokenizer import Tokenizer

LONGEST = 300          # characters of a message she will read
FORGET_AFTER = 30 * 60  # seconds of quiet before a channel's conversation starts fresh
PAUSE = 1.5            # seconds one person must wait between messages


def tidy(text: str, her_names: tuple[str, ...] = ()) -> str:
    """What a person typed, made ready for her: no pings to her, no links, one line."""
    for name in her_names:
        text = re.sub(rf"@?{re.escape(name)}\b[,:]?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|<a?:\w+:\d+>", " ", text)  # links and custom emoji
    return " ".join(text.split())[:LONGEST].strip()


class Room:
    """Every channel she talks in gets its own memory and its own mood."""

    def __init__(self, model: LiteGPT, tok: Tokenizer | None = None):
        self.model, self.tok = model, tok or Tokenizer()
        self.talks: dict[int, tuple[Excelsia, float]] = {}
        self.last: dict[int, float] = {}

    def too_soon(self, person: int, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if now - self.last.get(person, 0.0) < PAUSE:
            return True
        self.last[person] = now
        if len(self.last) > 5000:  # do not remember everyone forever
            self.last = {p: t for p, t in self.last.items() if now - t < 60}
        return False

    def reply(self, channel: int, text: str, now: float | None = None) -> tuple[str, str]:
        """Returns (her mood, what she says)."""
        now = time.time() if now is None else now
        talk, seen = self.talks.get(channel, (None, 0.0))
        if talk is None or now - seen > FORGET_AFTER:
            talk = Excelsia(self.model, self.tok)
        said = talk.reply(text)
        self.talks[channel] = (talk, now)
        return talk.mood.name, said
