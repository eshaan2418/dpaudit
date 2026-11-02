"""DP-SGD (Abadi et al. 2016) from scratch with torch.func per-sample gradients.

Each step: Poisson-sample a batch (every example independently with probability q), compute
per-example gradients with ``vmap(grad(loss))``, clip each to L2 norm ≤ C over all parameters,
sum, add N(0, σ²C²) noise per coordinate, divide by the *expected* batch size qN, and take an
SGD-with-momentum step. Poisson sampling plus division by the expected (not realised) batch
size is what the RDP accountant's subsampled Gaussian analysis assumes.

Setting ``noise_multiplier=None`` gives ordinary (non-private) minibatch SGD with the same
model, optimiser and schedule, as the ε = ∞ reference.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, vmap


class SmallCNN(nn.Module):
    """Tanh CNN for 28x28 grayscale (tanh trains better than ReLU under DP-SGD; Papernot
    et al. 2021)."""

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 8, 2, padding=3), nn.Tanh(), nn.MaxPool2d(2, 1),
            nn.Conv2d(16, 32, 4, 2), nn.Tanh(), nn.MaxPool2d(2, 1),
            nn.Flatten(), nn.Linear(32 * 4 * 4, 32), nn.Tanh(), nn.Linear(32, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                 nn.Tanh(), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DPConfig:
    epochs: int = 10
    sample_rate: float = 256 / 60000
    lr: float = 2.0
    momentum: float = 0.9
    max_grad_norm: float = 1.0
    noise_multiplier: float | None = 1.0  # None: non-private SGD
    seed: int = 0


def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, cfg: DPConfig,
          log_every: int = 0) -> dict:
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    n = len(x)
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    mom = {k: torch.zeros_like(v) for k, v in params.items()}
    loss_fn = nn.functional.cross_entropy

    def per_example_loss(p, b, xi, yi):  # type: ignore[no-untyped-def]
        out = functional_call(model, (p, b), (xi.unsqueeze(0),))
        return loss_fn(out, yi.unsqueeze(0))

    per_grad = vmap(grad(per_example_loss), in_dims=(None, None, 0, 0))
    steps = int(round(cfg.epochs / cfg.sample_rate))
    expected_b = cfg.sample_rate * n
    t0 = time.perf_counter()
    clip_frac = []
    for step in range(steps):
        lr = cfg.lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        mask = torch.rand(n, generator=g) < cfg.sample_rate
        idx = mask.nonzero().squeeze(1)
        if cfg.noise_multiplier is None:
            # non-private reference: fixed-size batches of the same expected size
            idx = torch.randint(0, n, (int(expected_b),), generator=g)
        total = {k: torch.zeros_like(v) for k, v in params.items()}
        for s in range(0, len(idx), 512):
            sub = idx[s : s + 512]
            gs = per_grad(params, buffers, x[sub], y[sub])
            if cfg.noise_multiplier is not None:
                norms = torch.sqrt(sum((v.reshape(len(sub), -1) ** 2).sum(1) for v in
                                       gs.values()))
                factor = (cfg.max_grad_norm / (norms + 1e-12)).clamp(max=1.0)
                clip_frac.append(float((factor < 1).float().mean()))
                for k, v in gs.items():
                    total[k] += (v * factor.view(-1, *([1] * (v.dim() - 1)))).sum(0)
            else:
                for k, v in gs.items():
                    total[k] += v.sum(0)
        for k in params:
            if cfg.noise_multiplier is not None:
                noise = torch.randn(params[k].shape, generator=g) * (
                    cfg.noise_multiplier * cfg.max_grad_norm)
                gk = (total[k] + noise) / expected_b
            else:
                gk = total[k] / len(idx)
            mom[k] = cfg.momentum * mom[k] + gk
            params[k] = params[k] - lr * mom[k]
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1}/{steps}", flush=True)
    with torch.no_grad():
        for k, p in model.named_parameters():
            p.copy_(params[k])
    return {"steps": steps, "wall_s": time.perf_counter() - t0,
            "clip_fraction": float(np.mean(clip_frac)) if clip_frac else 0.0}


@torch.no_grad()
def per_example_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), 2048):
        out.append(nn.functional.cross_entropy(model(x[i : i + 2048]), y[i : i + 2048],
                                               reduction="none"))
    return torch.cat(out).numpy()


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(x), 2048):
        correct += int((model(x[i : i + 2048]).argmax(1) == y[i : i + 2048]).sum())
    return correct / len(x)
