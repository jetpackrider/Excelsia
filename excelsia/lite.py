"""Excelsia without PyTorch, for small servers.

PyTorch is hundreds of megabytes; she is not. This file does the same arithmetic
with plain numpy, which is enough to run her in about 200 MB of memory. It reads
a file made by `python -m excelsia.export`, and it keeps the one rule that
matters: a word she is not allowed to say has its score set to minus infinity
before she chooses, so she cannot say it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mood import Mood
from .tokenizer import SPEAK_PUNCT, Tokenizer


def _erf(x: np.ndarray) -> np.ndarray:
    """The error function, to about 7 decimal places (Abramowitz and Stegun 7.1.26)."""
    sign, x = np.sign(x), np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
               + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-x * x))


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def _norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    x = x - x.mean(-1, keepdims=True)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * weight + bias


class LiteGPT:
    """The same network as model.GPT, in numpy, for generating only."""

    def __init__(self, path: str | Path):
        with np.load(path) as f:
            self.config = json.loads(str(f["config"]))
            self.w = {k: f[k].astype(np.float32) for k in f.files if k != "config"}
        self.heads, self.block = self.config["n_head"], self.config["block_size"]
        self.layers = self.config["n_layer"]

    def _block(self, i: int, x: np.ndarray, cache: list) -> np.ndarray:
        """Run block i on new positions `x` (T, C), using and extending its cache of
        the keys and values from earlier positions so they are never recomputed."""
        w, p = self.w, f"blocks.{i}."
        T, C = x.shape
        H, D = self.heads, C // self.heads
        q, k, v = np.split(_norm(x, w[p + "ln1.weight"], w[p + "ln1.bias"]) @ w[p + "qkv.weight"].T, 3, -1)
        q, k, v = (t.reshape(T, H, D).transpose(1, 0, 2) for t in (q, k, v))  # (H, T, D)
        if cache[i] is not None:
            k, v = np.concatenate([cache[i][0], k], 1), np.concatenate([cache[i][1], v], 1)
        cache[i] = (k, v)
        att = q @ k.transpose(0, 2, 1) / np.sqrt(D)  # (H, T, total)
        total = k.shape[1]
        if T > 1:  # each new position may only look at itself and what came before
            ahead = np.arange(total)[None, :] > (total - T + np.arange(T))[:, None]
            att = np.where(ahead[None], -np.inf, att)
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        y = (att @ v).transpose(1, 0, 2).reshape(T, C)
        x = x + y @ w[p + "proj.weight"].T
        h = _norm(x, w[p + "ln2.weight"], w[p + "ln2.bias"])
        h = _gelu(h @ w[p + "mlp.0.weight"].T + w[p + "mlp.0.bias"])
        return x + h @ w[p + "mlp.2.weight"].T + w[p + "mlp.2.bias"]

    def _scores(self, ids: list[int], start: int, cache: list) -> np.ndarray:
        """Feed token ids sitting at positions start, start+1, ... Returns the scores for
        whatever comes after the last of them."""
        w = self.w
        x = w["tok_emb.weight"][ids] + w["pos_emb.weight"][start:start + len(ids)]
        for i in range(self.layers):
            x = self._block(i, x, cache)
        last = _norm(x[-1], w["ln_f.weight"], w["ln_f.bias"])
        return w["tok_emb.weight"] @ last  # the output head shares the word table

    def generate(self, ids: list[int], allowed: np.ndarray, stop: int, max_new: int = 48,
                 temperature: float = 0.8, top_k: int = 40, not_first: list[int] | None = None,
                 rng: np.random.Generator | None = None) -> list[int]:
        rng = rng or np.random.default_rng()
        ids = ids[-(self.block - max_new):]
        cache: list = [None] * self.layers
        scores = self._scores(ids, 0, cache)
        out: list[int] = []
        for step in range(max_new):
            s = scores.astype(np.float64) / max(temperature, 1e-5)
            s[~allowed] = -np.inf
            if step == 0:
                s[[stop] + (not_first or [])] = -np.inf  # open with a word
            if top_k:
                kth = np.sort(s)[-min(top_k, int(allowed.sum()))]
                s[s < kth] = -np.inf
            p = np.exp(s - s.max())
            nxt = int(rng.choice(len(p), p=p / p.sum()))
            if nxt == stop:
                break
            out.append(nxt)
            scores = self._scores([nxt], len(ids) + len(out) - 1, cache)
        return out


class Excelsia:
    """One conversation with her: what has been said, and how she feels about it.
    Several conversations can share one LiteGPT, and may share one Mood.

    `remember` is how many earlier exchanges she keeps in view. None means as many
    as fit. A small number keeps her from being steered by her own old replies."""

    def __init__(self, model: LiteGPT, tok: Tokenizer | None = None, mood: Mood | None = None,
                 remember: int | None = None):
        self.model, self.tok = model, tok or Tokenizer()
        self.allowed = np.array(self.tok.speak_mask)
        self.punct = [self.tok.stoi[p] for p in SPEAK_PUNCT]
        self.mood = mood or Mood()
        self.remember = remember
        self.turns: list[list[int]] = []  # each finished exchange, as token ids

    def reply(self, text: str, temperature: float = 0.8, max_new: int = 48,
              rng: np.random.Generator | None = None) -> str:
        tok = self.tok
        mood = self.mood.hear(text)
        asked = [tok.user] + tok.encode(text) + [tok.mood(mood), tok.excelsia]
        before = self.turns if self.remember is None else self.turns[len(self.turns) - self.remember:]
        prompt = [t for turn in before for t in turn] + asked
        prompt = prompt[-(self.model.block - max_new):]  # she only sees what fits
        out = self.model.generate(prompt, self.allowed, tok.end, max_new, temperature,
                                  not_first=self.punct, rng=rng)
        self.turns = (self.turns + [asked + out + [tok.end]])[-12:]
        return tok.decode(out)
