"""Talk to Excelsia in the terminal.

    python -m excelsia.chat

Her mood lives out here, not inside the model: kind words lift it, unkind words
sink it, and it drifts back toward calm over a few messages. The current mood is
handed to the model as a tag before each reply.
"""
from __future__ import annotations

import argparse
import re

import torch

from .tokenizer import SPEAK_PUNCT, Tokenizer
from .train import WORK, load, pick_device

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


class Excelsia:
    def __init__(self, ckpt: str | None = None, device: str | None = None):
        self.tok = Tokenizer()
        self.device = device or pick_device()
        path = WORK / (ckpt or "excelsia.pt")
        self.model, self.info = load(path, self.device)
        self.allowed = torch.tensor(self.tok.speak_mask)
        self.punct = [self.tok.stoi[p] for p in SPEAK_PUNCT]
        self.mood = Mood()
        self.history: list[int] = []

    def reply(self, text: str, temperature: float = 0.8, max_new: int = 48) -> str:
        tok = self.tok
        mood = self.mood.hear(text)
        self.history += [tok.user] + tok.encode(text) + [tok.mood(mood), tok.excelsia]
        room = self.model.config.block_size - max_new
        self.history = self.history[-room:]  # she only remembers what fits
        idx = torch.tensor([self.history], device=self.device)
        out = self.model.generate(idx, self.allowed, tok.end, max_new, temperature,
                                  not_first=self.punct)
        self.history += out + [tok.end]
        return tok.decode(out)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m excelsia.chat")
    p.add_argument("--ckpt", help="checkpoint file inside work/ (default excelsia.pt)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--say", action="append", metavar="TEXT",
                   help="say this, print her reply and exit (can be repeated)")
    a = p.parse_args()
    try:
        bot = Excelsia(a.ckpt)
    except FileNotFoundError:
        raise SystemExit("No trained model found in work/. See the README for the steps.")
    if bot.info.get("stage") != "finetune":
        print("(This checkpoint has only learned English, not how to be Excelsia yet.)")
    if a.say:
        for text in a.say:
            said = bot.reply(text, a.temperature)
            print(f"you> {text}\nexcelsia [{bot.mood.name}]> {said}\n")
        return
    print("Excelsia is here. Type to talk, or press Ctrl-C to go.\n")
    try:
        while True:
            text = input("you> ").strip()
            if text:
                said = bot.reply(text, a.temperature)
                print(f"excelsia [{bot.mood.name}]> {said}\n")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
