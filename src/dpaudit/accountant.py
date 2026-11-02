"""Rényi-DP accountant for the Poisson-subsampled Gaussian mechanism (DP-SGD).

For sampling rate q, noise multiplier σ and an integer order α ≥ 2, one step of the subsampled
Gaussian mechanism has RDP (Mironov, Talwar, Zhang 2019, Sec. 3.3, integer α)

    ε_α = 1/(α−1) · log Σ_{k=0}^{α} C(α,k) (1−q)^{α−k} q^k exp((k² − k) / (2σ²)).

RDP composes additively over T steps. Conversion to (ε, δ)-DP uses the improved bound of
Balle et al. (2020) / Canonne et al. (2020), the same one used by Opacus:

    ε = min_α [ T·ε_α + log((α−1)/α) − (log δ + log α)/(α−1) ].

Everything is computed in log-space to avoid overflow at large α.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, logsumexp

ORDERS = tuple(range(2, 257))


def _log_comb(n: int, k: np.ndarray) -> np.ndarray:
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def rdp_step(q: float, sigma: float, alpha: int) -> float:
    """RDP of one subsampled Gaussian step at integer order alpha."""
    if q == 0:
        return 0.0
    if q == 1.0:
        return alpha / (2 * sigma**2)
    k = np.arange(alpha + 1, dtype=np.float64)
    terms = (_log_comb(alpha, k) + (alpha - k) * math.log1p(-q) + k * math.log(q)
             + (k * k - k) / (2 * sigma**2))
    return float(logsumexp(terms) / (alpha - 1))


def epsilon(q: float, sigma: float, steps: int, delta: float,
            orders: tuple[int, ...] = ORDERS) -> tuple[float, int]:
    """(ε, best order) after `steps` steps."""
    best = (math.inf, orders[0])
    for a in orders:
        rdp = steps * rdp_step(q, sigma, a)
        eps = rdp + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        if eps < best[0]:
            best = (eps, a)
    return max(best[0], 0.0), best[1]


def calibrate_sigma(target_eps: float, q: float, steps: int, delta: float,
                    lo: float = 0.3, hi: float = 50.0, tol: float = 1e-3) -> float:
    """Smallest noise multiplier achieving ε ≤ target (bisection; ε is decreasing in σ)."""
    if epsilon(q, hi, steps, delta)[0] > target_eps:
        raise ValueError("target epsilon not reachable with sigma <= hi")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if epsilon(q, mid, steps, delta)[0] > target_eps:
            lo = mid
        else:
            hi = mid
    return hi
