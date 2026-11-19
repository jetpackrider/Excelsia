"""Turn words up or down by editing her weights directly. No training involved.

    python -m excelsia.tweak --down people body --by 2
    python -m excelsia.tweak --up forest waters trees --by 3 --out excelsia-forest.pt
    python -m excelsia.chat --ckpt excelsia-forest.pt

How it works: the very last thing her network does is add a fixed list of 384
numbers (a "bias") before it scores every word. Nudging that list raises or
lowers a word's score everywhere, in every conversation. This works out the
smallest nudge that moves the words you name by the amount you ask while
disturbing every other word as little as possible. 384 numbers out of about 18
million change; nothing else is touched.

It changes which words she reaches for, not what she knows. Turn a favourite
word down and she will say something nearby instead, which is not always better.
The original file is never modified: the result is saved under a new name.
"""
from __future__ import annotations

import argparse
import json

import torch

from .tokenizer import Tokenizer
from .train import DATA, WORK, load, save


def word_weights(tok: Tokenizer, chats=DATA / "chats.jsonl") -> torch.Tensor:
    """How much each word matters to her, judged by how often the practice chats use it.
    Words she says a lot are the ones the edit works hardest to leave alone."""
    count = torch.full((len(tok.itos),), 0.5)
    if chats.exists():
        for line in chats.read_text(encoding="utf-8").splitlines():
            if line.strip():
                for turn in json.loads(line)["turns"]:
                    for i in tok.encode(turn["reply"]):
                        count[i] += 1
    count = count * torch.tensor(tok.speak_mask)
    return count / count.sum()


@torch.no_grad()
def nudge(model, tok: Tokenizer, changes: dict[str, float], weights: torch.Tensor) -> dict:
    """Shift the final bias so each word in `changes` moves by its amount and everything
    else moves as little as possible. Returns a small report."""
    table = model.tok_emb.weight.detach().float().cpu()  # one row per word; also the output head
    ids = [tok.stoi[w] for w in changes]
    want = torch.tensor([float(v) for v in changes.values()])
    spread = (table * weights[:, None]).T @ table + 1e-4 * torch.eye(table.shape[1])
    rows = table[ids]
    solved = torch.linalg.solve(spread, rows.T)
    delta = solved @ torch.linalg.solve(rows @ solved, want)
    model.ln_f.bias += delta.to(model.ln_f.bias.device, model.ln_f.bias.dtype)
    moved = table @ delta
    others = moved.clone()
    others[ids] = 0
    others = others * torch.tensor(tok.speak_mask)
    worst = torch.topk(others.abs(), 5).indices.tolist()
    return {"changed": delta.numel(), "total": sum(p.numel() for p in model.parameters()),
            "moved": {w: round(moved[i].item(), 2) for w, i in zip(changes, ids)},
            "side_effects": [(tok.itos[i], round(others[i].item(), 2)) for i in worst]}


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m excelsia.tweak")
    p.add_argument("--down", nargs="*", default=[], metavar="WORD", help="words to make rarer")
    p.add_argument("--up", nargs="*", default=[], metavar="WORD", help="words to make more common")
    p.add_argument("--by", type=float, default=2.0,
                   help="how hard to push. 1 is gentle, 2 is clear, 3 is strong (default 2)")
    p.add_argument("--ckpt", default="excelsia.pt", help="file inside work/ to start from")
    p.add_argument("--out", default="excelsia-tweaked.pt", help="file inside work/ to save as")
    a = p.parse_args()

    tok = Tokenizer()
    changes = {w.lower(): -a.by for w in a.down} | {w.lower(): a.by for w in a.up}
    if not changes:
        raise SystemExit("Name at least one word with --down or --up.")
    cannot = [w for w in changes if w not in tok.speak_words]
    if cannot:
        raise SystemExit(f"She cannot say: {', '.join(cannot)}. Only words on her list can be tuned.")
    if a.out == a.ckpt:
        raise SystemExit("Pick a different --out name so the original is kept.")
    source = WORK / a.ckpt
    if not source.exists():
        raise SystemExit(f"No {a.ckpt} in work/.")

    model, info = load(source, "cpu")
    report = nudge(model, tok, changes, word_weights(tok))
    save(model, WORK / a.out, info.get("step", 0), info.get("stage", "finetune"))
    print(f"Changed {report['changed']} of {report['total']:,} numbers.")
    for word, amount in report["moved"].items():
        print(f"  {word}: {'up' if amount > 0 else 'down'} {abs(amount):g}")
    print("Words that moved most without being asked: "
          + ", ".join(f"{w} ({v:+g})" for w, v in report["side_effects"]))
    print(f"Saved work/{a.out}. Talk to her with:  python -m excelsia.chat --ckpt {a.out}")


if __name__ == "__main__":
    main()
