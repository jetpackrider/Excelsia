"""Train Excelsia.

    python -m excelsia.train stories            # download + tokenize TinyStories (once)
    python -m excelsia.train pretrain           # learn English from stories + the constitution
    python -m excelsia.train talk               # download + tokenize SODA conversations (once)
    python -m excelsia.train converse           # learn how conversations go, from SODA
    python -m excelsia.train finetune           # learn to be Excelsia from her practice chats

finetune can take several chat files, and a second set to learn from last:
    python -m excelsia.train finetune --chats data/chats.jsonl data/chats-talk.jsonl \
        --then data/chats-gpt.jsonl

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
# SODA: 1.5 million everyday two-person conversations (Kim et al. 2023, CC BY 4.0).
SODA_URL = "https://huggingface.co/datasets/allenai/soda/resolve/main/{split}.parquet"


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


def encode_dialogue(tok: Tokenizer, lines: list[str]) -> list[int]:
    """One SODA conversation in the same shape as her chats: the first speaker is the
    person, the second answers as she would. There is no mood: she learns that later."""
    ids = []
    for i, line in enumerate(lines):
        if i % 2 == 0:
            ids += [tok.user] + tok.encode(line)
        else:
            ids += [tok.excelsia] + tok.encode(line) + [tok.end]
    return ids


def usable_dialogue(lines: list[str], speakers: list[str]) -> bool:
    """Two people taking turns, at least two lines."""
    return (len(lines) >= 2 and len(lines) == len(speakers) and len(set(speakers)) == 2
            and all(a != b for a, b in zip(speakers, speakers[1:])))


def prepare_talk(n: int, source: str | None = None) -> None:
    """Download SODA's training conversations, keep the two-person ones that take turns,
    save them as token ids plus where each conversation starts."""
    import pyarrow.parquet as pq
    tok = Tokenizer()
    WORK.mkdir(exist_ok=True)
    path = Path(source) if source else WORK / "soda-train.parquet"
    if not path.exists():
        print("downloading SODA (about 690 MB)")
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(SODA_URL.format(split="train"), part)
        part.rename(path)
    ids, starts, kept, skipped = array("H"), array("I"), 0, 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=20000, columns=["dialogue", "speakers"]):
        for lines, speakers in zip(*(batch.column(c).to_pylist() for c in ("dialogue", "speakers"))):
            if not usable_dialogue(lines, speakers):
                skipped += 1
                continue
            starts.append(len(ids))
            ids.extend(encode_dialogue(tok, lines))
            kept += 1
            if kept % 100000 == 0:
                print(f"  {kept:,} conversations, {len(ids):,} tokens", flush=True)
            if 0 < n <= kept:
                break
        if 0 < n <= kept:
            break
    arr = np.frombuffer(ids, dtype=np.uint16)
    arr.tofile(WORK / "talk.bin")
    np.frombuffer(starts, dtype=np.uint32).tofile(WORK / "talk-starts.bin")
    unk = float((arr == tok.unk).mean())
    print(f"saved {kept:,} conversations ({skipped:,} skipped), {len(arr):,} tokens "
          f"({unk:.1%} unknown words)")


def constitution_ids(tok: Tokenizer) -> np.ndarray:
    text = (DATA / "ny_constitution.txt").read_text(encoding="utf-8")
    return np.array(tok.encode(text), dtype=np.uint16)


def lm_batcher(sources: list[tuple], block: int, batch: int, unk: int, device: str):
    """Random windows for plain next-word training. sources = [(tokens, weight), ...] or
    [(tokens, weight, starts), ...]: with starts, every window begins at one of them (the
    start of a conversation, the way her real chats begin) and runs on into the next."""
    arrays = [s[0] for s in sources]
    weights = [s[1] for s in sources]
    begins = []
    for s in sources:
        st = s[2] if len(s) > 2 else None
        begins.append(None if st is None else st[st < len(s[0]) - block - 1])

    def get():
        xs, ys = [], []
        for _ in range(batch):
            k = random.choices(range(len(arrays)), weights)[0]
            a, st = arrays[k], begins[k]
            i = int(st[random.randrange(len(st))]) if st is not None else random.randrange(0, len(a) - block - 1)
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
        try:
            chat = json.loads(line)
        except ValueError:  # a line cut off mid-write
            dropped += 1
            continue
        if all(tok.can_say(t["reply"])[0] and t["reply"].strip() for t in chat["turns"]):
            rows.append(encode_chat(tok, chat, block))
        else:
            dropped += 1
    print(f"{len(rows):,} chats loaded, {dropped:,} dropped (words outside the list, or unreadable)")
    return rows


def pad_rows(picks, pad: int, device: str):
    width = max(len(x) for x, _ in picks)
    x = torch.full((len(picks), width), pad, dtype=torch.long)
    y = torch.full((len(picks), width), -100, dtype=torch.long)
    for r, (xi, yi) in enumerate(picks):
        x[r, :len(xi)] = torch.tensor(xi)
        y[r, :len(yi)] = torch.tensor(yi)
    return x.to(device), y.to(device)


def chat_batcher(rows, batch: int, pad: int, device: str, mix=None):
    """Random chats. mix = [(rows, weight), ...] draws each chat from one of several sets."""
    groups = mix or [(rows, 1.0)]
    sets, weights = [g for g, _ in groups], [w for _, w in groups]

    def get():
        picks = [random.choice(random.choices(sets, weights)[0]) for _ in range(batch)]
        return pad_rows(picks, pad, device)

    return get


def chat_batches(rows, batch: int, pad: int, device: str) -> list:
    """Every held-out chat, once, in fixed batches: the same yardstick at every check."""
    return [pad_rows(rows[i:i + batch], pad, device) for i in range(0, len(rows), batch)]


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
def evaluate(model: GPT, batches: list) -> float:
    model.eval()
    loss = sum(model(x, y)[1].item() for x, y in batches) / len(batches)
    model.train()
    return loss


def fixed(get_batch, rounds: int = 20) -> list:
    """Draw a held-out set once, so every check measures the same text."""
    return [get_batch() for _ in range(rounds)]


def fit(model: GPT, get_batch, val: list, steps: int, lr: float, out: Path, stage: str,
        minutes: float | None = None, eval_every: int = 250, no_amp: bool = False,
        patience: int = 0, start_is_saved: bool = False) -> float:
    """Train, checking held-out loss every eval_every steps and saving the best model to out.
    patience: stop after this many checks in a row without improvement (0 = never stop early).
    start_is_saved: out already holds the starting model, so only save if training beats it."""
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
    best, t0, seen, step, stale = float("inf"), time.time(), 0, 0, 0
    if start_is_saved:
        best = evaluate(model, val)
        print(f"  check: loss on held-out text {best:.3f} before this stage", flush=True)
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
                loss_now = evaluate(model, val)
                mark = ""
                if loss_now < best:
                    best, mark, stale = loss_now, "  (best so far, saved)", 0
                    save(model, out, step + 1, stage)
                else:
                    stale += 1
                print(f"  check: loss on held-out text {loss_now:.3f}{mark}", flush=True)
                if patience and stale >= patience:
                    print(f"  no better for {stale} checks in a row: stopping here")
                    break
            step += 1
    except KeyboardInterrupt:
        print("\nstopped by you")
    if best == float("inf"):
        save(model, out, step, stage)
    print(f"best held-out loss {best:.3f}, kept in {out.name}")
    return best


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
    val = fixed(lm_batcher([(stories[cut:], 1.0)], block, a.batch, tok.unk, device))
    fit(model, get, val, a.steps, a.lr, resume, "pretrain", a.minutes, no_amp=a.no_amp)


def cmd_converse(a: argparse.Namespace) -> None:
    """Between English and her own chats: read a great many ordinary conversations, so she
    learns that a reply answers what was just said. Stories and the constitution stay in
    the mix so she keeps what she already knows."""
    tok, device = Tokenizer(), pick_device()
    base = WORK / "pretrain.pt"
    if not base.exists():
        raise SystemExit("No pretrained model yet. Run:  python -m excelsia.train pretrain")
    if not (WORK / "talk.bin").exists():
        raise SystemExit("No conversations yet. Run:  python -m excelsia.train talk")
    model, _ = load(base, device)
    talk = np.fromfile(WORK / "talk.bin", dtype=np.uint16)
    starts = np.fromfile(WORK / "talk-starts.bin", dtype=np.uint32).astype(np.int64)
    stories = np.fromfile(WORK / "stories.bin", dtype=np.uint16) if (WORK / "stories.bin").exists() else None
    law = constitution_ids(tok)
    cut = int(starts[int(len(starts) * 0.98)])  # the last 2% of conversations are held out
    print(f"{model.n_params() / 1e6:.1f}M parameters on {device}; "
          f"{len(starts):,} conversations, {len(talk):,} conversation words")
    block = model.config.block_size
    mix = [(talk[:cut], 0.85, starts[starts < cut]), (law, 0.05)]
    mix.append((stories, 0.10) if stories is not None else (law, 0.10))
    get = lm_batcher(mix, block, a.batch, tok.unk, device)
    val = fixed(lm_batcher([(talk[cut:], 1.0, starts[starts >= cut] - cut)], block, a.batch, tok.unk, device))
    fit(model, get, val, a.steps, a.lr, WORK / "talker.pt", "converse", a.minutes, no_amp=a.no_amp)


def split_chats(paths: list[str], tok: Tokenizer, block: int, share: float):
    """Load and shuffle each file, hold out `share` of each (at least 10)."""
    train, held = [], []
    for path in paths:
        print(f"{path}: ", end="")
        rows = load_chats(Path(path), tok, block)
        random.Random(0).shuffle(rows)
        n_val = max(10, int(len(rows) * share))
        train += rows[n_val:]
        held += rows[:n_val]
    return train, held


def cmd_finetune(a: argparse.Namespace) -> None:
    tok, device = Tokenizer(), pick_device()
    base = Path(a.base) if a.base else next(
        (p for p in (WORK / "talker.pt", WORK / "pretrain.pt") if p.exists()), None)
    if base is None or not base.exists():
        raise SystemExit("No pretrained model yet. Run:  python -m excelsia.train pretrain")
    model, ckpt = load(base, device)
    print(f"starting from {base.name} ({ckpt.get('stage', '?')})")
    block, out = model.config.block_size, WORK / "excelsia.pt"
    train, held = split_chats(a.chats, tok, block, 0.05)
    if len(train) < 20:
        raise SystemExit("Need at least 20 usable chats to fine-tune.")
    print(f"== learning from {len(train):,} chats ({len(held):,} held out)")
    fit(model, chat_batcher(train, a.batch, tok.pad, device), chat_batches(held, a.batch, tok.pad, device),
        a.steps, a.lr, out, "finetune", a.minutes, eval_every=50, no_amp=a.no_amp, patience=a.patience)
    if not a.then:
        return
    # Last, the chats she should sound most like. Half of each batch still comes from the
    # first set so she does not forget it; the held-out yardstick is the new set alone.
    model, _ = load(out, device)
    last, last_held = split_chats(a.then, tok, block, 0.10)
    print(f"== then from {len(last):,} chats ({len(last_held):,} held out)")
    fit(model, chat_batcher(None, a.batch, tok.pad, device, mix=[(last, 0.5), (train, 0.5)]),
        chat_batches(last_held, a.batch, tok.pad, device), a.then_steps, a.lr / 2, out, "finetune",
        eval_every=10, no_amp=a.no_amp, patience=a.patience, start_is_saved=True)


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
    s = sub.add_parser("talk", help="download and tokenize SODA conversations")
    s.add_argument("--n", type=int, default=0, help="how many conversations to keep (0 = all)")
    s.add_argument("--file", help="use a local SODA train.parquet instead of downloading")
    s = sub.add_parser("converse", help="learn how conversations go")
    s.add_argument("--steps", type=int, default=12000)
    s.add_argument("--minutes", type=float, help="size the run to take about this long")
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--no-amp", action="store_true", help="turn off half-precision on NVIDIA GPUs")
    s = sub.add_parser("finetune", help="learn to be Excelsia")
    s.add_argument("--chats", nargs="+", default=[str(DATA / "chats.jsonl")])
    s.add_argument("--then", nargs="*", default=[], help="chats to learn from last")
    s.add_argument("--base", help="model to start from (default: work/talker.pt if it exists, "
                                  "else work/pretrain.pt)")
    s.add_argument("--steps", type=int, default=1500)
    s.add_argument("--then-steps", type=int, default=300)
    s.add_argument("--patience", type=int, default=6, help="stop after this many checks without improvement")
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
    elif a.cmd == "talk":
        prepare_talk(a.n, a.file)
    elif a.cmd == "converse":
        cmd_converse(a)
    else:
        cmd_finetune(a)


if __name__ == "__main__":
    main()
