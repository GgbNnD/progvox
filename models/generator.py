"""Detail synthesis generators for the ProGVC prototype."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class CNNGeneratorConfig:
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 32
    residual_blocks: int = 4
    residual_scale: float = 0.25


@dataclass(frozen=True)
class DiffusionConfig:
    image_channels: int = 3
    condition_channels: int = 3
    base_channels: int = 32
    timesteps: int = 16
    beta_start: float = 1e-4
    beta_end: float = 0.02


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.net(x))


class DetailSynthesisCNN(nn.Module):
    """A light residual CNN refiner conditioned on tokenizer reconstruction."""

    def __init__(self, config: CNNGeneratorConfig | None = None) -> None:
        super().__init__()
        self.config = config or CNNGeneratorConfig()
        blocks = [nn.Conv2d(self.config.in_channels, self.config.base_channels, 3, padding=1), nn.SiLU(inplace=True)]
        blocks.extend(ResidualBlock(self.config.base_channels) for _ in range(self.config.residual_blocks))
        blocks.extend(
            [
                nn.Conv2d(self.config.base_channels, self.config.base_channels, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.config.base_channels, self.config.out_channels, 3, padding=1),
            ]
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, condition: Tensor) -> Tensor:
        residual = torch.tanh(self.net(condition)) * self.config.residual_scale
        return (condition + residual).clamp(0, 1)


class PatchDiscriminator(nn.Module):
    """PatchGAN-style discriminator for conditional generator training."""

    def __init__(self, in_channels: int = 6, base_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(4, base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, 3, padding=1),
        )

    def forward(self, condition: Tensor, image: Tensor) -> Tensor:
        return self.net(torch.cat([condition, image], dim=1))


class TimeEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.SiLU(inplace=True),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        half = self.channels // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if embedding.size(1) < self.channels:
            embedding = F.pad(embedding, (0, self.channels - embedding.size(1)))
        return self.mlp(embedding)


class TinyDenoiser(nn.Module):
    """Small denoising CNN used by the conditional diffusion prototype."""

    def __init__(self, image_channels: int = 3, condition_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.time = TimeEmbedding(base_channels)
        self.input = nn.Conv2d(image_channels + condition_channels, base_channels, 3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(base_channels) for _ in range(3)])
        self.output = nn.Conv2d(base_channels, image_channels, 3, padding=1)

    def forward(self, noisy: Tensor, condition: Tensor, timesteps: Tensor) -> Tensor:
        h = self.input(torch.cat([noisy, condition], dim=1))
        temb = self.time(timesteps).view(timesteps.size(0), -1, 1, 1)
        h = h + temb
        for block in self.blocks:
            h = block(h)
        return self.output(h)


class TinyConditionalDiffusion(nn.Module):
    """Conditional DDPM-style refiner with a short sampler for live demos."""

    def __init__(self, config: DiffusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or DiffusionConfig()
        self.denoiser = TinyDenoiser(
            image_channels=self.config.image_channels,
            condition_channels=self.config.condition_channels,
            base_channels=self.config.base_channels,
        )
        betas = torch.linspace(self.config.beta_start, self.config.beta_end, self.config.timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)

    def q_sample(self, target: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        sqrt_alpha = self.alpha_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1.0 - self.alpha_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha * target + sqrt_one_minus * noise

    def training_loss(self, condition: Tensor, target: Tensor) -> Tensor:
        batch = target.size(0)
        timesteps = torch.randint(0, self.config.timesteps, (batch,), device=target.device)
        noise = torch.randn_like(target)
        noisy = self.q_sample(target, timesteps, noise)
        predicted = self.denoiser(noisy, condition, timesteps)
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def sample(self, condition: Tensor, steps: int = 4) -> Tensor:
        sample = condition + 0.05 * torch.randn_like(condition)
        schedule = torch.linspace(self.config.timesteps - 1, 0, steps, device=condition.device).long()
        for timestep in schedule:
            t = torch.full((condition.size(0),), int(timestep.item()), device=condition.device, dtype=torch.long)
            noise = self.denoiser(sample, condition, t)
            alpha = self.alphas[t].view(-1, 1, 1, 1)
            alpha_bar = self.alpha_cumprod[t].view(-1, 1, 1, 1)
            beta = self.betas[t].view(-1, 1, 1, 1)
            sample = (sample - beta / (1.0 - alpha_bar).sqrt() * noise) / alpha.sqrt()
            if timestep.item() > 0:
                sample = sample + beta.sqrt() * torch.randn_like(sample)
        return sample.clamp(0, 1)


def build_generator(name: str, base_channels: int = 32) -> nn.Module:
    """Factory used by tests and training scripts."""

    if name in {"cnn", "pure_cnn", "cnn_gan"}:
        return DetailSynthesisCNN(CNNGeneratorConfig(base_channels=base_channels))
    if name in {"diffusion", "tiny_diffusion"}:
        return TinyConditionalDiffusion(DiffusionConfig(base_channels=base_channels))
    raise ValueError(f"Unknown generator: {name}")
