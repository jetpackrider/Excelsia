"""A small GPT. Nothing exotic: token + position embeddings, pre-norm blocks, tied output head."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

SIZES = {  # name: (layers, heads, width)
    "tiny": (4, 4, 192),
    "small": (6, 8, 256),
    "medium": (8, 8, 384),
}


@dataclass
class Config:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_head, self.dropout = c.n_head, c.dropout
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(c.n_embd, 4 * c.n_embd), nn.GELU(), nn.Linear(4 * c.n_embd, c.n_embd))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))
        return x + self.drop(self.mlp(self.ln2(x)))


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.config = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids. targets: same shape, -100 where no loss is wanted."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, allowed: torch.Tensor, stop: int,
                 max_new: int = 48, temperature: float = 0.8, top_k: int = 40,
                 not_first: list[int] | None = None) -> list[int]:
        """Sample a reply. `allowed` is a (vocab,) bool tensor on the CPU; any token where
        it is False gets a score of -inf before sampling, so it can never be chosen."""
        self.eval()
        out: list[int] = []
        for step in range(max_new):
            logits, _ = self(idx[:, -self.config.block_size:])
            logits = logits[0, -1].float().cpu() / max(temperature, 1e-5)  # sample on CPU
            logits = logits.masked_fill(~allowed, float("-inf"))
            if step == 0:
                logits[[stop] + (not_first or [])] = float("-inf")  # open with a word
            if top_k:
                kth = torch.topk(logits, min(top_k, int(allowed.sum()))).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            if nxt.item() == stop:
                break
            out.append(nxt.item())
            idx = torch.cat([idx, nxt.view(1, 1).to(idx.device)], dim=1)
        return out
