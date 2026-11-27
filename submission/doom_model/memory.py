"""Learned attention over recurrent latent slots; no coordinates, maps or action rules.

Attention/residual-gating reference (MIT), independently implemented:
https://github.com/MarcoMeter/episodic-transformer-memory-ppo/blob/acc5cb2e0cdc87bd7b5ce857d270a72a66146e0b/transformer.py
Unlike that episodic transformer, these are learned recurrent slots, not a history window.
"""
from __future__ import annotations

import torch
from torch import nn


class LatentSlotMemory(nn.Module):
    """Read/write [N, slots, dim] using only encoder embeddings [N, embed].

    Slot embeddings are learned addresses, not episode state. Episode contents
    start at zero and remain differentiable within each rollout sequence.
    """

    def __init__(self, embed: int, slots: int, dim: int):
        super().__init__()
        self.slot_embeddings = nn.Parameter(torch.empty(slots, dim))
        nn.init.normal_(self.slot_embeddings, std=dim ** -0.5)
        self.write_query = nn.Linear(embed, dim)
        self.write_key = nn.Linear(dim, dim, bias=False)
        self.erase = nn.Linear(embed, dim)
        self.write_gate = nn.Linear(embed, dim)
        self.write_value = nn.Linear(embed, dim)
        self.read_query = nn.Linear(embed, dim)
        self.read_key = nn.Linear(dim, dim, bias=False)
        self.read_value = nn.Linear(dim, dim, bias=False)
        self.read_projection = nn.Linear(dim, embed, bias=False)
        # A base policy is unchanged at migration; this projection learns first.
        nn.init.zeros_(self.read_projection.weight)

    @staticmethod
    def _attention(query, keys):
        scores = (keys * query[:, None, :]).sum(-1) * (query.shape[-1] ** -0.5)
        return scores.softmax(dim=-1)[..., None]

    def forward(self, x, slots):
        addressed = slots + self.slot_embeddings
        weights = self._attention(self.write_query(x), self.write_key(addressed))
        erase = self.erase(x).sigmoid()[:, None, :]
        write = self.write_gate(x).sigmoid()[:, None, :] * self.write_value(x).tanh()[:, None, :]
        slots = slots * (1 - weights * erase) + weights * write
        # Reading after writing also trains the residual projection on the first step.
        weights = self._attention(self.read_query(x), self.read_key(slots + self.slot_embeddings))
        read = (weights * self.read_value(slots)).sum(1)
        return self.read_projection(read), slots
