"""Her mood. It lives outside the model: kind words lift it, unkind words sink it, and it
drifts back toward calm over a few messages. Nothing here needs PyTorch."""
from __future__ import annotations

import re

KIND = set("""thank thanks love lovely like nice good great kind sweet friend friends happy
glad wonderful beautiful pretty best awesome amazing proud please sorry welcome hug cute
smart brave care gentle fun funny yay""".split())
UNKIND = set("""hate stupid dumb ugly bad worst shut boring useless idiot terrible awful
annoying horrible gross mean liar silly wrong hush lame pathetic worthless""".split())
LEAVING = set("""bye goodbye leaving leave gone goodnight later farewell""".split())


class Mood:
    """One number from -1 (hurt) to +1 (delighted), plus a flag for being left alone."""

    def __init__(self) -> None:
        self.level, self.left = 0.0, False

    def hear(self, text: str) -> str:
        words = re.findall(r"[a-z']+", text.lower())
        self.level *= 0.7  # feelings fade
        self.level += 0.45 * sum(w in KIND for w in words)
        self.level -= 0.55 * sum(w in UNKIND for w in words)
        self.level = max(-1.0, min(1.0, self.level))
        self.left = any(w in LEAVING for w in words)
        return self.name

    @property
    def name(self) -> str:
        if self.left and self.level < 0.5:
            return "alone"
        if self.level > 0.6:
            return "pleasure"
        if self.level > 0.2:
            return "grateful"
        if self.level < -0.6:
            return "offense"
        if self.level < -0.2:
            return "pain"
        return "peace"
