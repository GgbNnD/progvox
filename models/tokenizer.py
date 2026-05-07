"""Multi-scale residual tokenizer for the ProGVC prototype.

This module implements the phase-2.1 tokenizer deliverable as a compact,
deterministic PyTorch component:

1. Build a coarse-to-fine image pyramid.
2. At each scale, quantize the residual against the previous reconstructed scale.
3. Return discrete token maps plus enough metadata to reconstruct from any
   transmitted prefix of layers.

The default codebooks are uniform RGB residual grids. They are usable without
training for quick experiments, but can also be made learnable and optimized in
future training scripts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class TokenizerConfig:
    """Configuration for :class:`MultiScaleResidualTokenizer`."""

    levels: int = 4
    codebook_size: int = 512
    channels: int = 3
    residual_range: float = 1.0
    residual_ranges: tuple[float, ...] | None = None
    max_token_resolution: int | None = 32
    interpolation: str = "bicubic"
    learnable_codebooks: bool = False
    clamp_output: bool = True


@dataclass
class TokenizerOutput:
    """Discrete token maps and reconstructions produced by the tokenizer."""

    tokens: list[Tensor]
    reconstructions: list[Tensor]
    final_reconstruction: Tensor
    shapes: list[tuple[int, int]]
    bits_per_token: int

    def rate_bits(self, max_level: int | None = None) -> int:
        """Return the number of token bits needed through ``max_level``."""

        if max_level is None:
            max_level = len(self.tokens) - 1
        if max_level < 0:
            return 0
        if max_level >= len(self.tokens):
            raise ValueError("max_level exceeds available token maps")
        return int(
            sum(token.numel() for token in self.tokens[: max_level + 1])
            * self.bits_per_token
        )

    def estimate_kbps(self, fps: float, max_level: int | None = None) -> float:
        """Estimate single-frame token bitrate at a given frame rate."""

        return self.rate_bits(max_level=max_level) * fps / 1000.0


def _make_uniform_residual_codebook(
    codebook_size: int,
    channels: int,
    residual_range: float,
) -> Tensor:
    """Create a deterministic Cartesian residual codebook."""

    if codebook_size <= 1:
        raise ValueError("codebook_size must be greater than 1")
    if channels <= 0:
        raise ValueError("channels must be positive")
    levels = math.ceil(codebook_size ** (1.0 / channels))
    values = torch.linspace(-residual_range, residual_range, levels)
    mesh = torch.meshgrid(*([values] * channels), indexing="ij")
    codebook = torch.stack([axis.reshape(-1) for axis in mesh], dim=-1)
    if codebook.size(0) < codebook_size:
        pad = torch.zeros(codebook_size - codebook.size(0), channels)
        codebook = torch.cat([codebook, pad], dim=0)
    return codebook[:codebook_size].contiguous()


class VectorQuantizer(nn.Module):
    """Nearest-neighbour vector quantizer for residual tensors."""

    def __init__(
        self,
        codebook_size: int,
        channels: int,
        residual_range: float = 1.0,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        codebook = _make_uniform_residual_codebook(codebook_size, channels, residual_range)
        if learnable:
            self.codebook = nn.Parameter(codebook)
        else:
            self.register_buffer("codebook", codebook)

    @property
    def codebook_size(self) -> int:
        return int(self.codebook.size(0))

    @property
    def channels(self) -> int:
        return int(self.codebook.size(1))

    def forward(self, residual: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize a residual tensor.

        Args:
            residual: Tensor in ``B x C x H x W`` layout.

        Returns:
            ``(indices, quantized_residual)`` where indices are ``B x H x W``.
        """

        if residual.dim() != 4:
            raise ValueError("residual must be a 4D BxCxHxW tensor")
        if residual.size(1) != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {residual.size(1)}")

        flat = residual.permute(0, 2, 3, 1).reshape(-1, self.channels)
        codebook = self.codebook.to(dtype=flat.dtype, device=flat.device)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = distances.argmin(dim=1)
        quantized = F.embedding(indices, codebook).view(
            residual.size(0), residual.size(2), residual.size(3), self.channels
        )
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        if self.training:
            quantized = residual + (quantized - residual).detach()
        return indices.view(residual.size(0), residual.size(2), residual.size(3)), quantized

    def dequantize(self, indices: Tensor) -> Tensor:
        """Convert a ``B x H x W`` token map back to residual vectors."""

        if indices.dim() != 3:
            raise ValueError("indices must be a 3D BxHxW tensor")
        if indices.min().item() < 0 or indices.max().item() >= self.codebook_size:
            raise ValueError("token index out of codebook range")
        codebook = self.codebook.to(device=indices.device)
        residual = F.embedding(indices.long(), codebook)
        return residual.permute(0, 3, 1, 2).contiguous()


class MultiScaleResidualTokenizer(nn.Module):
    """Coarse-to-fine residual tokenizer with truncation-aware reconstruction."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TokenizerConfig()
        if self.config.levels <= 0:
            raise ValueError("levels must be positive")
        if self.config.residual_ranges is not None and len(self.config.residual_ranges) != self.config.levels:
            raise ValueError("residual_ranges length must match levels")
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(
                    self.config.codebook_size,
                    self.config.channels,
                    residual_range=self._residual_range_for_level(level),
                    learnable=self.config.learnable_codebooks,
                )
                for level in range(self.config.levels)
            ]
        )
        self.bits_per_token = math.ceil(math.log2(self.config.codebook_size))

    def _residual_range_for_level(self, level: int) -> float:
        if self.config.residual_ranges is not None:
            return float(self.config.residual_ranges[level])
        return float(self.config.residual_range / (2**level))

    def scale_shapes(self, height: int, width: int) -> list[tuple[int, int]]:
        """Return ``levels`` coarse-to-fine spatial shapes for an input size."""

        if self.config.max_token_resolution is None:
            max_height, max_width = height, width
        else:
            max_height = min(height, self.config.max_token_resolution)
            max_width = min(width, self.config.max_token_resolution)

        shapes: list[tuple[int, int]] = []
        for level in range(self.config.levels):
            divisor = 2 ** (self.config.levels - level - 1)
            shapes.append(
                (
                    max(1, math.ceil(max_height / divisor)),
                    max(1, math.ceil(max_width / divisor)),
                )
            )
        return shapes

    def _resize(self, x: Tensor, shape: tuple[int, int]) -> Tensor:
        if tuple(x.shape[-2:]) == shape:
            return x
        kwargs = {"size": shape, "mode": self.config.interpolation}
        if self.config.interpolation in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        return F.interpolate(x, **kwargs)

    def encode(self, x: Tensor) -> TokenizerOutput:
        """Encode a batch of RGB frames into multi-scale residual token maps."""

        if x.dim() != 4:
            raise ValueError("input must be a 4D BxCxHxW tensor")
        if x.size(1) != self.config.channels:
            raise ValueError(f"expected {self.config.channels} channels, got {x.size(1)}")

        original_shape = tuple(x.shape[-2:])
        x = x.clamp(0, 1)
        tokens: list[Tensor] = []
        reconstructions: list[Tensor] = []
        previous: Tensor | None = None
        shapes = self.scale_shapes(*original_shape)

        for level, shape in enumerate(shapes):
            target = self._resize(x, shape)
            prior = torch.zeros_like(target) if previous is None else self._resize(previous, shape)
            residual = target - prior
            indices, quantized_residual = self.quantizers[level](residual)
            reconstruction = prior + quantized_residual
            if self.config.clamp_output:
                reconstruction = reconstruction.clamp(0, 1)
            tokens.append(indices)
            full_size_reconstruction = self._resize(reconstruction, original_shape)
            if self.config.clamp_output:
                full_size_reconstruction = full_size_reconstruction.clamp(0, 1)
            reconstructions.append(full_size_reconstruction)
            previous = reconstruction

        return TokenizerOutput(
            tokens=tokens,
            reconstructions=reconstructions,
            final_reconstruction=reconstructions[-1],
            shapes=shapes,
            bits_per_token=self.bits_per_token,
        )

    forward = encode

    def reconstruct(
        self,
        tokens: Sequence[Tensor],
        output_shape: tuple[int, int],
        max_level: int | None = None,
    ) -> Tensor:
        """Reconstruct an RGB tensor from a transmitted token prefix."""

        if not tokens:
            raise ValueError("at least one token map is required")
        if max_level is None:
            max_level = len(tokens) - 1
        if max_level < 0 or max_level >= len(tokens) or max_level >= len(self.quantizers):
            raise ValueError("invalid max_level")

        previous: Tensor | None = None
        for level in range(max_level + 1):
            residual = self.quantizers[level].dequantize(tokens[level])
            prior = torch.zeros_like(residual) if previous is None else self._resize(previous, residual.shape[-2:])
            previous = prior + residual
            if self.config.clamp_output:
                previous = previous.clamp(0, 1)

        assert previous is not None
        reconstruction = self._resize(previous, output_shape)
        return reconstruction.clamp(0, 1) if self.config.clamp_output else reconstruction

    def level_bit_counts(self, tokens: Iterable[Tensor]) -> list[int]:
        """Return per-level token bit counts."""

        return [int(token.numel() * self.bits_per_token) for token in tokens]
