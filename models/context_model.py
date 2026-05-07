"""Autoregressive token context model for ProGVC token maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ContextModelConfig:
    codebook_size: int = 512
    levels: int = 4
    max_sequence_length: int = 2048
    d_model: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.0


class TokenContextTransformer(nn.Module):
    """Small causal Transformer over flattened multi-scale token maps."""

    def __init__(self, config: ContextModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ContextModelConfig()
        self.token_embedding = nn.Embedding(self.config.codebook_size, self.config.d_model)
        self.level_embedding = nn.Embedding(self.config.levels, self.config.d_model)
        self.position_embedding = nn.Embedding(self.config.max_sequence_length, self.config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.d_model * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.config.num_layers)
        self.norm = nn.LayerNorm(self.config.d_model)
        self.output = nn.Linear(self.config.d_model, self.config.codebook_size)

    def forward(self, tokens: Tensor, level_ids: Tensor | None = None) -> Tensor:
        """Return logits for each token position.

        Args:
            tokens: ``B x N`` token ids.
            level_ids: optional ``B x N`` or ``N`` level ids.
        """

        if tokens.dim() != 2:
            raise ValueError("tokens must be a BxN tensor")
        if tokens.size(1) > self.config.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")
        if tokens.min().item() < 0 or tokens.max().item() >= self.config.codebook_size:
            raise ValueError("token id out of range")

        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device).unsqueeze(0).expand(batch, -1)
        if level_ids is None:
            level_ids = torch.zeros_like(tokens)
        elif level_ids.dim() == 1:
            level_ids = level_ids.unsqueeze(0).expand(batch, -1)
        if level_ids.shape != tokens.shape:
            raise ValueError("level_ids must match tokens shape")

        hidden = (
            self.token_embedding(tokens)
            + self.level_embedding(level_ids.clamp(0, self.config.levels - 1))
            + self.position_embedding(positions)
        )
        mask = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(hidden, mask=mask)
        return self.output(self.norm(hidden))

    def next_token_loss(self, tokens: Tensor, level_ids: Tensor | None = None) -> Tensor:
        """Compute next-token cross entropy over a full sequence."""

        if tokens.size(1) < 2:
            raise ValueError("need at least two tokens for next-token loss")
        logits = self(tokens[:, :-1], None if level_ids is None else level_ids[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))

    @torch.no_grad()
    def greedy_predict(
        self,
        prefix: Tensor,
        level_prefix: Tensor,
        target_level_ids: Tensor,
        fallback_token_ids: Tensor | None = None,
        max_autoregressive_steps: int = 256,
    ) -> Tensor:
        """Predict missing tokens, using fallback ids beyond the AR budget.

        This keeps the prototype fast for dense token maps while preserving a
        real autoregressive interface for later training.
        """

        if prefix.dim() != 2:
            raise ValueError("prefix must be BxN")
        batch = prefix.size(0)
        generated = prefix
        levels = level_prefix
        predictions = []
        budget = min(int(max_autoregressive_steps), int(target_level_ids.numel()))
        for index in range(budget):
            logits = self(generated, levels)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            predictions.append(next_token)
            next_level = target_level_ids[index].view(1, 1).expand(batch, 1).to(prefix.device)
            generated = torch.cat([generated, next_token], dim=1)
            levels = torch.cat([levels, next_level], dim=1)

        remaining = target_level_ids.numel() - budget
        if remaining > 0:
            if fallback_token_ids is None:
                fallback = torch.zeros(batch, remaining, device=prefix.device, dtype=torch.long)
            else:
                fallback = fallback_token_ids[budget : budget + remaining].view(1, -1).expand(batch, -1).to(prefix.device)
            predictions.append(fallback)
        return torch.cat(predictions, dim=1) if predictions else torch.empty(batch, 0, device=prefix.device, dtype=torch.long)


def flatten_token_maps(tokens: Sequence[Tensor]) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
    """Flatten ``B x H x W`` token maps into ``B x N`` ids and level ids."""

    if not tokens:
        raise ValueError("tokens cannot be empty")
    batch = tokens[0].size(0)
    flat_tokens = []
    flat_levels = []
    shapes = []
    for level, token_map in enumerate(tokens):
        if token_map.dim() != 3:
            raise ValueError("each token map must be BxHxW")
        if token_map.size(0) != batch:
            raise ValueError("all token maps must share the batch dimension")
        shapes.append(tuple(token_map.shape[-2:]))
        flat = token_map.reshape(batch, -1)
        flat_tokens.append(flat)
        flat_levels.append(torch.full_like(flat, level))
    return torch.cat(flat_tokens, dim=1).long(), torch.cat(flat_levels, dim=1).long(), shapes


def unflatten_token_maps(flat: Tensor, shapes: Sequence[tuple[int, int]]) -> list[Tensor]:
    """Convert flattened token ids back to token maps."""

    if flat.dim() != 2:
        raise ValueError("flat must be BxN")
    maps = []
    offset = 0
    for height, width in shapes:
        count = height * width
        maps.append(flat[:, offset : offset + count].reshape(flat.size(0), height, width))
        offset += count
    if offset != flat.size(1):
        raise ValueError("flat token length does not match shapes")
    return maps


def fill_missing_with_fallback(
    received_tokens: Sequence[Tensor],
    target_shapes: Sequence[tuple[int, int]],
    fallback_token_ids: Sequence[int],
    device: torch.device | None = None,
) -> list[Tensor]:
    """Append fallback token maps for missing scales."""

    if not received_tokens:
        raise ValueError("at least one received token map is required")
    batch = received_tokens[0].size(0)
    device = device or received_tokens[0].device
    output = [token.to(device) for token in received_tokens]
    for level in range(len(output), len(target_shapes)):
        height, width = target_shapes[level]
        fill_value = int(fallback_token_ids[level])
        output.append(torch.full((batch, height, width), fill_value, device=device, dtype=torch.long))
    return output
