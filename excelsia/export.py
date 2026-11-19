"""Pack her into one small file that runs without PyTorch.

    python -m excelsia.export                      # work/excelsia.pt -> bot/excelsia.npz
    python -m excelsia.export --web                # work/excelsia.pt -> web/excelsia.bin
    python -m excelsia.export --ckpt excelsia-tweaked.pt

For the Discord bot the numbers are stored at half precision, which halves the
file (about 35 MB for the medium model). For the website they are stored as
whole numbers from -127 to 127 with one scale per row, which quarters it (about
18 MB) so it downloads quickly. Both change her scores by far less than the
randomness she already speaks with. The web file also carries her vocabulary,
so the page needs nothing else.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .train import DATA, WORK

BOT = DATA.parent / "bot"
WEB = DATA.parent / "web"


def export(source: Path, out: Path) -> None:
    ckpt = torch.load(source, map_location="cpu", weights_only=True)
    arrays = {name: t.numpy().astype(np.float16) for name, t in ckpt["model"].items()
              if name != "head.weight"}  # the head is the word table again; no need for two
    out.parent.mkdir(exist_ok=True)
    np.savez(out, config=json.dumps(ckpt["config"]), **arrays)
    print(f"saved {out.name}  ({out.stat().st_size / 1e6:.0f} MB, "
          f"{sum(a.size for a in arrays.values()):,} numbers)")


def export_web(source: Path, out: Path) -> None:
    """One file for the browser: a JSON header (settings, vocabulary, where each tensor
    sits), then the numbers. Matrices are int8 with a float32 scale per row; the small
    one-dimensional tensors stay float32."""
    ckpt = torch.load(source, map_location="cpu", weights_only=True)
    vocab = json.loads((DATA / "vocab.json").read_text(encoding="utf-8"))
    tensors, blobs, offset = [], [], 0

    def add(blob: bytes) -> int:
        nonlocal offset
        blob += b"\0" * (-len(blob) % 4)  # keep everything lined up on 4 bytes
        blobs.append(blob)
        at, offset = offset, offset + len(blob)
        return at

    for name, t in ckpt["model"].items():
        if name == "head.weight":
            continue
        w = t.numpy().astype(np.float32)
        if w.ndim == 2:
            scale = np.maximum(np.abs(w).max(1) / 127.0, 1e-12).astype(np.float32)
            q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
            tensors.append({"name": name, "shape": list(w.shape), "kind": "i8",
                            "scale": add(scale.tobytes()), "data": add(q.tobytes())})
        else:
            tensors.append({"name": name, "shape": list(w.shape), "kind": "f32",
                            "data": add(w.tobytes())})
    header = json.dumps({"config": ckpt["config"], "vocab": vocab, "tensors": tensors},
                        ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 4)
    out.parent.mkdir(exist_ok=True)
    with out.open("wb") as f:
        f.write(b"EXW1" + len(header).to_bytes(4, "little") + header)
        for blob in blobs:
            f.write(blob)
    print(f"saved {out.name}  ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m excelsia.export")
    p.add_argument("--ckpt", default="excelsia.pt", help="file inside work/ to pack")
    p.add_argument("--web", action="store_true", help="pack for the website instead of the bot")
    p.add_argument("--out", help="where to save (default bot/excelsia.npz, or web/excelsia.bin)")
    a = p.parse_args()
    source = WORK / a.ckpt
    if not source.exists():
        raise SystemExit(f"No {a.ckpt} in work/.")
    if a.web:
        export_web(source, Path(a.out or WEB / "excelsia.bin"))
    else:
        export(source, Path(a.out or BOT / "excelsia.npz"))


if __name__ == "__main__":
    main()
