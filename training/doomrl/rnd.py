"""Observation-only Random Network Distillation, training only.

Algorithm references (independent implementation):
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py
https://github.com/openai/random-network-distillation

Unlike the original continuing intrinsic task, this experiment uses episodic
intrinsic GAE. Terminal/reset images receive no bonus: the official wrapper has
no fresh terminal image. No coordinates, object rules or actor embeddings enter
the novelty networks. Observation statistics are fixed for each PPO rollout.
"""
from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn

from doom_model.artifacts import check_finite
from doom_model.features import PIX_H, PIX_W


@dataclasses.dataclass(frozen=True)
class RNDConfig:
    coefficient: float
    gamma: float
    gae_lambda: float
    learning_rate: float
    feature_dim: int
    batch_size: int
    epochs: int
    observation_clip: float
    variance_floor: float
    max_grad_norm: float
    value_coef: float

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name in ("feature_dim", "batch_size", "epochs"):
                if type(value) is not int or value <= 0:
                    raise ValueError(f"RND {field.name} must be a positive integer")
            elif type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"RND {field.name} must be finite and positive")
        if self.gamma > 1 or self.gae_lambda > 1:
            raise ValueError("RND discounts must be <= 1")


class RunningMoments(nn.Module):
    """Parallel population-moment updates, with no invented prior samples."""

    def __init__(self, shape=()):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("variance", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, samples):
        if not samples.numel():
            return
        samples = samples.to(dtype=torch.float64)
        if not torch.isfinite(samples).all():
            raise ValueError("nonfinite running-moment input")
        mean = samples.mean(0)
        variance = samples.var(0, unbiased=False)
        n = samples.shape[0]
        total = self.count + n
        delta = mean - self.mean
        combined = (self.variance * self.count + variance * n
                    + delta.square() * self.count * n / total)
        self.mean.add_(delta * n / total)
        self.variance.copy_(combined / total)
        self.count.copy_(total)


class NoveltyEncoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.LeakyReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            width = self.conv(torch.zeros(1, 4, PIX_H, PIX_W)).shape[-1]
        self.projection = nn.Linear(width, feature_dim)

    def forward(self, image):
        return self.projection(self.conv(image))


class RND(nn.Module):
    def __init__(self, config: RNDConfig, hidden: int, device):
        super().__init__()
        self.config = config
        self.target = NoveltyEncoder(config.feature_dim)
        self.predictor = nn.Sequential(
            NoveltyEncoder(config.feature_dim), nn.ReLU(),
            nn.Linear(config.feature_dim, config.feature_dim),
        )
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.observations = RunningMoments((4,))
        self.returns = RunningMoments()
        self.target.requires_grad_(False)
        self.to(device)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=config.learning_rate)
        self.discounted = None

    @torch.no_grad()
    def observe(self, pixels):
        """Call before the rollout and after fitting, never during its PPO epochs."""
        pixels = pixels.reshape(-1, 4, PIX_H, PIX_W)
        # Chunk to keep moment estimation's float64 workspace bounded.
        for chunk in pixels.split(self.config.batch_size):
            x = chunk.float().div(255).permute(0, 2, 3, 1).reshape(-1, 4)
            self.observations.update(x)

    def normalize(self, pixels):
        x = pixels.float() / 255
        mean = self.observations.mean.float()[None, :, None, None]
        scale = self.observations.variance.float().clamp_min(self.config.variance_floor).sqrt()
        return ((x - mean) / scale[None, :, None, None]).clamp(
            -self.config.observation_clip, self.config.observation_clip)

    @torch.no_grad()
    def reward(self, pixels, done):
        """Prediction error only for genuine nonterminal next observations."""
        result = torch.zeros(done.shape, device=pixels.device)
        valid = ~done.bool()
        if valid.any():
            x = self.normalize(pixels[valid])
            result[valid] = (self.predictor(x) - self.target(x)).square().mean(-1)
        return result

    @torch.no_grad()
    def normalize_rewards(self, rewards, dones):
        if self.discounted is None:
            self.discounted = torch.zeros(rewards.shape[1], device=rewards.device)
        accumulated = []
        for reward, done in zip(rewards, dones):
            self.discounted = self.discounted * self.config.gamma + reward
            accumulated.append(self.discounted.clone())
            self.discounted = self.discounted * (1 - done)
        self.returns.update(torch.stack(accumulated).flatten())
        scale = self.returns.variance.float().clamp_min(self.config.variance_floor).sqrt()
        return rewards / scale

    def fit(self, pixels, dones):
        pixels = pixels.flatten(0, 1)[~dones.flatten().bool()]
        if not len(pixels):
            return 0.0
        losses = []
        for _ in range(self.config.epochs):
            indices = torch.randperm(len(pixels), device=pixels.device)
            for batch in indices.split(self.config.batch_size):
                x = self.normalize(pixels[batch])
                with torch.no_grad():
                    target = self.target(x)
                loss = (self.predictor(x) - target).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite RND predictor loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.predictor.parameters(), self.config.max_grad_norm,
                                         error_if_nonfinite=True)
                self.optimizer.step()
                losses.append(float(loss.detach()))
        self.observe(pixels)
        return sum(losses) / len(losses)

    def save(self):
        check_finite(self.state_dict())
        return {"config": dataclasses.asdict(self.config), "model": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "resume_semantics": "statistics restored; discounted episode accumulators reset"}

    def restore(self, saved):
        if saved["config"] != dataclasses.asdict(self.config):
            raise ValueError("resume RND configuration mismatch")
        check_finite(saved["model"])
        self.load_state_dict(saved["model"], strict=True)
        self.optimizer.load_state_dict(saved["optimizer"])
        self.discounted = None  # PPO also starts fresh simulator episodes.
