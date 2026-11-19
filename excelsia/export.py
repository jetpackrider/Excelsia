"""Pack her into one small file the Discord bot can run without PyTorch.

    python -m excelsia.export                      # work/excelsia.pt -> bot/excelsia.npz
    python -m excelsia.export --ckpt excelsia-tweaked.pt

The numbers are stored at half precision, which halves the file (about 35 MB for
the medium model) and changes her scores by far less than the randomness she
already speaks with.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .train import DATA, WORK

BOT = DATA.parent / "bot"


def export(source: Path, out: Path) -> None:
    ckpt = torch.load(source, map_location="cpu", weights_only=True)
    arrays = {name: t.numpy().astype(np.float16) for name, t in ckpt["model"].items()
              if name != "head.weight"}  # the head is the word table again; no need for two
    out.parent.mkdir(exist_ok=True)
    np.savez(out, config=json.dumps(ckpt["config"]), **arrays)
    print(f"saved {out.name}  ({out.stat().st_size / 1e6:.0f} MB, "
          f"{sum(a.size for a in arrays.values()):,} numbers)")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m excelsia.export")
    p.add_argument("--ckpt", default="excelsia.pt", help="file inside work/ to pack")
    p.add_argument("--out", default=str(BOT / "excelsia.npz"))
    a = p.parse_args()
    source = WORK / a.ckpt
    if not source.exists():
        raise SystemExit(f"No {a.ckpt} in work/.")
    export(source, Path(a.out))


if __name__ == "__main__":
    main()
