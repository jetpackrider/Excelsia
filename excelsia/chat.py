"""Talk to Excelsia in the terminal.

    python -m excelsia.chat

Her mood lives out here, not inside the model: kind words lift it, unkind words
sink it, and it drifts back toward calm over a few messages. The current mood is
handed to the model as a tag before each reply.
"""
from __future__ import annotations

import argparse

import torch

from .mood import Mood
from .tokenizer import SPEAK_PUNCT, Tokenizer
from .train import WORK, load, pick_device

REPEAT = 0.0  # off unless asked for; try 1 to make her vary her words more
SMALL_WORDS = """the a an of and or is are be shall to in not no i my it this that by for with as
at on from but all new york""".split()


class Excelsia:
    def __init__(self, ckpt: str | None = None, device: str | None = None, remember: int = 0):
        self.tok = Tokenizer()
        self.device = device or pick_device()
        path = WORK / (ckpt or "excelsia.pt")
        self.model, self.info = load(path, self.device)
        self.allowed = torch.tensor(self.tok.speak_mask)
        self.punct = [self.tok.stoi[p] for p in SPEAK_PUNCT]
        self.mood = Mood()
        # How many earlier exchanges she keeps in view. None by default: shown her own
        # earlier replies, she copies them instead of answering what was just said.
        self.remember = remember
        self.turns: list[list[int]] = []
        self.said: list[int] = []  # her own recent words, so she can tire of them
        # Little words she may repeat freely. Everything else wears out for a while.
        self.small = set(self.punct) | {self.tok.stoi[w] for w in SMALL_WORDS if w in self.tok.stoi}

    def reply(self, text: str, temperature: float = 0.8, max_new: int = 48,
              repeat: float = REPEAT) -> str:
        tok = self.tok
        mood = self.mood.hear(text)
        asked = [tok.user] + tok.encode(text) + [tok.mood(mood), tok.excelsia]
        before = self.turns[len(self.turns) - self.remember:] if self.remember else []
        prompt = ([t for turn in before for t in turn] + asked)[-(self.model.config.block_size - max_new):]
        idx = torch.tensor([prompt], device=self.device)
        out = self.model.generate(idx, self.allowed, tok.end, max_new, temperature,
                                  not_first=self.punct, recent=self.said, repeat=repeat,
                                  exempt=self.small)
        self.said = (self.said + out)[-60:]  # roughly her last three or four replies
        self.turns = (self.turns + [asked + out + [tok.end]])[-12:]
        return tok.decode(out)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m excelsia.chat")
    p.add_argument("--ckpt", help="checkpoint file inside work/ (default excelsia.pt)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--remember", type=int, default=0,
                   help="earlier exchanges she keeps in view (default 0: she answers better without)")
    p.add_argument("--repeat", type=float, default=REPEAT,
                   help="how quickly she tires of her own words (0 = never)")
    p.add_argument("--say", action="append", metavar="TEXT",
                   help="say this, print her reply and exit (can be repeated)")
    a = p.parse_args()
    try:
        bot = Excelsia(a.ckpt, remember=a.remember)
    except FileNotFoundError:
        raise SystemExit("No trained model found in work/. See the README for the steps.")
    if bot.info.get("stage") != "finetune":
        print("(This checkpoint has only learned English, not how to be Excelsia yet.)")
    if a.say:
        for text in a.say:
            said = bot.reply(text, a.temperature, repeat=a.repeat)
            print(f"you> {text}\nexcelsia [{bot.mood.name}]> {said}\n")
        return
    print("Excelsia is here. Type to talk, or press Ctrl-C to go.\n")
    try:
        while True:
            text = input("you> ").strip()
            if text:
                said = bot.reply(text, a.temperature, repeat=a.repeat)
                print(f"excelsia [{bot.mood.name}]> {said}\n")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
