"""The part of the Discord bot that has nothing to do with Discord: who is talking to
her where, and what she says back. Kept separate so it can be tested on its own."""
from __future__ import annotations

import re
import time

from excelsia.lite import Excelsia, LiteGPT
from excelsia.mood import Mood
from excelsia.tokenizer import Tokenizer

LONGEST = 300          # characters of a message she will read
FORGET_AFTER = 30 * 60  # seconds of quiet before a thread, or her mood, starts fresh
REMEMBER = 2           # earlier exchanges with the same person she keeps in view
PAUSE = 1.5            # seconds one person must wait between messages


def tidy(text: str, her_names: tuple[str, ...] = ()) -> str:
    """What a person typed, made ready for her: no pings to her, no links, one line."""
    for name in her_names:
        text = re.sub(rf"@?{re.escape(name)}\b[,:]?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|<a?:\w+:\d+>", " ", text)  # links and custom emoji
    return " ".join(text.split())[:LONGEST].strip()


class Room:
    """Who she is talking to, and where. Each person in a channel has their own thread
    with her, so a follow-up connects to that person's last message and not to someone
    else's. Her mood is shared by the whole channel: one rude guest sours her for all."""

    def __init__(self, model: LiteGPT, tok: Tokenizer | None = None):
        self.model, self.tok = model, tok or Tokenizer()
        self.talks: dict[tuple[int, int], tuple[Excelsia, float]] = {}
        self.moods: dict[int, tuple[Mood, float]] = {}
        self.last: dict[int, float] = {}

    def too_soon(self, person: int, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if now - self.last.get(person, 0.0) < PAUSE:
            return True
        self.last[person] = now
        if len(self.last) > 5000:  # do not remember everyone forever
            self.last = {p: t for p, t in self.last.items() if now - t < 60}
        return False

    def reply(self, channel: int, person: int, text: str, now: float | None = None) -> tuple[str, str]:
        """Returns (her mood, what she says)."""
        now = time.time() if now is None else now
        mood, felt = self.moods.get(channel, (None, 0.0))
        if mood is None or now - felt > FORGET_AFTER:
            mood = Mood()
        talk, seen = self.talks.get((channel, person), (None, 0.0))
        if talk is None or now - seen > FORGET_AFTER:
            talk = Excelsia(self.model, self.tok, remember=REMEMBER)
        talk.mood = mood
        said = talk.reply(text)
        self.moods[channel] = (mood, now)
        self.talks[(channel, person)] = (talk, now)
        if len(self.talks) > 500:  # let go of threads that have gone quiet
            self.talks = {k: v for k, v in self.talks.items() if now - v[1] <= FORGET_AFTER}
        return mood.name, said
