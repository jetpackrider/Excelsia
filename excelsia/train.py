"""Train Excelsia.

    python -m excelsia.train stories            # download + tokenize TinyStories (once)
    python -m excelsia.train pretrain           # learn English from stories + the constitution
    python -m excelsia.train finetune --chats data/chats.jsonl   # learn to be Excelsia

Everything it writes goes in work/ (ignored by git).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.request
from array import array
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .model import GPT, SIZES, Config
from .tokenizer import DATA, Tokenizer

WORK = DATA.parent / "work"
STORIES_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
               "TinyStoriesV2-GPT4-train.txt")
SEP = "<|endoftext|>"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    return "cpu"


# ---------------------------------------------------------------- data

def prepare_stories(n: int, source: str | None = None) -> None:
    """Stream TinyStories, keep the first n stories, save them as token ids."""
    tok, ids, kept, buf = Tokenizer(), array("H"), 0, []
    WORK.mkdir(exist_ok=True)
    stream = open(source, "rb") if source else urllib.request.urlopen(STORIES_URL)
    with stream:
        for raw in stream:
            line = raw.decode("utf-8", errors="ignore")
            if SEP not in line:
                buf.append(line)
                continue
            ids.append(tok.story)
            ids.extend(tok.encode(" ".join(buf)))
            buf, kept = [], kept + 1
            if kept % 20000 == 0:
                print(f"  {kept:,} stories, {len(ids):,} tokens")
            if 0 < n <= kept:
                break
    arr = np.frombuffer(ids, dtype=np.uint16)
    arr.tofile(WORK / "stories.bin")
    unk = float((arr == tok.unk).mean())
    print(f"saved {kept:,} stories, {len(arr):,} tokens ({unk:.1%} unknown words)")


def constitution_ids(tok: Tokenizer) -> np.ndarray:
    text = (DATA / "ny_constitution.txt").read_text(encoding="utf-8")
    return np.array(tok.encode(text), dtype=np.uint16)


def lm_batcher(sources: list[tuple[np.ndarray, float]], block: int, batch: int,
               unk: int, device: str):
    """Random windows for plain next-word training. sources = [(tokens, weight), ...]."""
    arrays = [a for a, _ in sources]
    weights = [w for _, w in sources]

    def get():
        xs, ys = [], []
        for _ in range(batch):
            a = random.choices(arrays, weights)[0]
            i = random.randrange(0, len(a) - block - 1)
            chunk = a[i:i + block + 1].astype(np.int64)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        x, y = torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
        y[y == unk] = -100  # never train her to produce "unknown word"
        return x.to(device), y.to(device)

    return get


def encode_chat(tok: Tokenizer, chat: dict, block: int) -> tuple[list[int], list[int]]:
    """One chat -> (input ids, targets). Loss is only taken on Excelsia's own words."""
    ids, is_reply = [], []
    for turn in chat["turns"]:
        head = [tok.user] + tok.encode(turn["user"]) + [tok.mood(turn["mood"]), tok.excelsia]
        reply = tok.encode(turn["reply"]) + [tok.end]
        ids += head + reply
        is_reply += [False] * len(head) + [True] * len(reply)
    ids, is_reply = ids[-(block + 1):], is_reply[-(block + 1):]
    x = ids[:-1]
    y = [t if keep else -100 for t, keep in zip(ids[1:], is_reply[1:])]
    return x, y


def load_chats(path: Path, tok: Tokenizer, block: int):
    """Read chats, dropping any whose reply contains a word she is not allowed to say."""
    rows, dropped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chat = json.loads(line)
        if all(tok.can_say(t["reply"])[0] and t["reply"].strip() for t in chat["turns"]):
            rows.append(encode_chat(tok, chat, block))
        else:
            dropped += 1
    print(f"{len(rows):,} chats loaded, {dropped:,} dropped for using words outside the list")
    return rows


def chat_batcher(rows, batch: int, pad: int, device: str):
    def get():
        picks = random.sample(rows, min(batch, len(rows)))
        width = max(len(x) for x, _ in picks)
        x = torch.full((len(picks), width), pad, dtype=torch.long)
        y = torch.full((len(picks), width), -100, dtype=torch.long)
        for r, (xi, yi) in enumerate(picks):
            x[r, :len(xi)] = torch.tensor(xi)
            y[r, :len(yi)] = torch.tensor(yi)
        return x.to(device), y.to(device)

    return get


# ---------------------------------------------------------------- loop

def save(model: GPT, path: Path, step: int, stage: str) -> None:
    WORK.mkdir(exist_ok=True)
    torch.save({"config": asdict(model.config), "model": model.state_dict(),
                "step": step, "stage": stage}, path)


def load(path: Path, device: str) -> tuple[GPT, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


@torch.no_grad()
def evaluate(model: GPT, get_batch, rounds: int = 20) -> float:
    model.eval()
    loss = sum(model(*get_batch())[1].item() for _ in range(rounds)) / rounds
    model.train()
    return loss


def fit(model: GPT, get_batch, get_val, steps: int, lr: float, out: Path, stage: str,
        minutes: float | None = None, eval_every: int = 250, no_amp: bool = False) -> None:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    plain = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": plain, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))
    total = steps
    # On NVIDIA GPUs, half-precision maths roughly doubles the speed. Not used on Macs.
    fast = next(model.parameters()).is_cuda and not no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=fast)

    def lr_at(s: int) -> float:
        warmup = max(20, total // 20)
        if s < warmup:
            return lr * (s + 1) / warmup
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))

    model.train()
    best, t0, seen, step = float("inf"), time.time(), 0, 0
    try:
        while step < total:
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            x, y = get_batch()
            with torch.autocast("cuda", dtype=torch.float16, enabled=fast):
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            seen += x.numel()
            elapsed = time.time() - t0
            if minutes and step == 30:  # now we know the speed: size the run to the budget
                fits = int(minutes * 60 / (elapsed / 31))
                if fits < total:
                    total = max(fits, step + 2)
                    print(f"  fitting the run into {minutes:g} minutes: {total} steps", flush=True)
            if step % 50 == 0 or step == total - 1:
                left = elapsed / (step + 1) * (total - step - 1) / 60
                print(f"step {step:>6}/{total}  loss {loss.item():.3f}  "
                      f"{seen / max(elapsed, 1e-9):,.0f} words/sec  about {left:.0f} min left",
                      flush=True)
            if (step + 1) % eval_every == 0 or step == total - 1:
                val = evaluate(model, get_val)
                mark = ""
                if val < best:
                    best, mark = val, "  (best so far, saved)"
                    save(model, out, step + 1, stage)
                print(f"  check: loss on held-out text {val:.3f}{mark}", flush=True)
            step += 1
    except KeyboardInterrupt:
        print("\nstopped by you")
    if best == float("inf"):
        save(model, out, step, stage)
    print(f"saved {out.relative_to(DATA.parent)}")


# ---------------------------------------------------------------- commands

def cmd_pretrain(a: argparse.Namespace) -> None:
    tok, device = Tokenizer(), pick_device()
    path = WORK / "stories.bin"
    if not path.exists():
        raise SystemExit("No stories yet. Run:  python -m excelsia.train stories")
    stories = np.fromfile(path, dtype=np.uint16)
    law = constitution_ids(tok)
    cut = int(len(stories) * 0.98)
    n_layer, n_head, n_embd = SIZES[a.size]
    resume = WORK / "pretrain.pt"
    if a.resume and resume.exists():
        model, _ = load(resume, device)
    else:
        model = GPT(Config(len(tok), a.block, n_layer, n_head, n_embd)).to(device)
    print(f"{model.n_params() / 1e6:.1f}M parameters on {device}; "
          f"{len(stories):,} story words, {len(law):,} constitution words")
    block = model.config.block_size
    get = lm_batcher([(stories[:cut], 0.9), (law, 0.1)], block, a.batch, tok.unk, device)
    val = lm_batcher([(stories[cut:], 1.0)], block, a.batch, tok.unk, device)
    fit(model, get, val, a.steps, a.lr, resume, "pretrain", a.minutes, no_amp=a.no_amp)


def cmd_finetune(a: argparse.Namespace) -> None:
    tok, device = Tokenizer(), pick_device()
    base = WORK / "pretrain.pt"
    if not base.exists():
        raise SystemExit("No pretrained model yet. Run:  python -m excelsia.train pretrain")
    model, _ = load(base, device)
    rows = load_chats(Path(a.chats), tok, model.config.block_size)
    if len(rows) < 20:
        raise SystemExit("Need at least 20 usable chats to fine-tune.")
    random.Random(0).shuffle(rows)
    n_val = max(10, len(rows) // 20)
    get = chat_batcher(rows[n_val:], a.batch, tok.pad, device)
    val = chat_batcher(rows[:n_val], a.batch, tok.pad, device)
    fit(model, get, val, a.steps, a.lr, WORK / "excelsia.pt", "finetune", a.minutes,
        eval_every=100, no_amp=a.no_amp)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m excelsia.train")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stories", help="download and tokenize TinyStories")
    s.add_argument("--n", type=int, default=200_000, help="how many stories to keep (0 = all)")
    s.add_argument("--file", help="use a local TinyStories .txt instead of downloading")
    s = sub.add_parser("pretrain", help="learn English")
    s.add_argument("--size", choices=SIZES, default="small")
    s.add_argument("--steps", type=int, default=6000)
    s.add_argument("--minutes", type=float, help="size the run to take about this long")
    s.add_argument("--batch", type=int, default=48)
    s.add_argument("--block", type=int, default=256)
    s.add_argument("--lr", type=float, default=6e-4)
    s.add_argument("--resume", action="store_true", help="continue from work/pretrain.pt")
    s.add_argument("--no-amp", action="store_true", help="turn off half-precision on NVIDIA GPUs")
    s = sub.add_parser("finetune", help="learn to be Excelsia")
    s.add_argument("--chats", default=str(DATA / "chats.jsonl"))
    s.add_argument("--steps", type=int, default=1500)
    s.add_argument("--minutes", type=float)
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--no-amp", action="store_true", help="turn off half-precision on NVIDIA GPUs")
    a = p.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    if a.cmd == "stories":
        prepare_stories(a.n, a.file)
    elif a.cmd == "pretrain":
        cmd_pretrain(a)
    else:
        cmd_finetune(a)


if __name__ == "__main__":
    main()
