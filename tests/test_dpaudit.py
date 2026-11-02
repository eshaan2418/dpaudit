import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from dpaudit.accountant import ORDERS, calibrate_sigma, epsilon, rdp_step
from dpaudit.audit import eps_lower_bound, one_run_audit, roc_auc, tpr_at_fpr
from dpaudit.dpsgd import MLP, DPConfig, train


@settings(max_examples=40, deadline=None)
@given(st.floats(0.001, 0.2), st.floats(0.5, 5.0), st.integers(1, 5000))
def test_accountant_matches_opacus(q, sigma, steps):
    rdp_mod = pytest.importorskip("opacus.accountants.analysis.rdp")
    rdp = rdp_mod.compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=list(ORDERS))
    theirs, _ = rdp_mod.get_privacy_spent(orders=list(ORDERS), rdp=rdp, delta=1e-5)
    ours, _ = epsilon(q, sigma, steps, 1e-5)
    assert abs(ours - theirs) <= 1e-8 * max(1.0, abs(theirs))


def test_rdp_full_batch_is_gaussian():
    assert abs(rdp_step(1.0, 2.0, 8) - 8 / 8) < 1e-12


def test_known_value_abadi_setting():
    # q = 0.01, sigma = 1.1, 10k steps, delta = 1e-5 (the setting of Abadi et al. 2016)
    eps, _ = epsilon(0.01, 1.1, 10_000, 1e-5)
    assert 5.6 < eps < 5.7


def test_calibration_hits_target():
    s = calibrate_sigma(2.0, 0.005, 2000, 1e-5)
    assert epsilon(0.005, s, 2000, 1e-5)[0] <= 2.0
    assert epsilon(0.005, s - 0.01, 2000, 1e-5)[0] > 2.0 - 1e-6


def test_per_sample_clipping_bounds_sensitivity():
    """Removing one example changes a (noise-free) DP-SGD step by at most 2C·lr/N."""
    torch.manual_seed(0)
    x = torch.randn(64, 5) * 100
    y = torch.randint(0, 2, (64,))
    m1, m2 = MLP(5), MLP(5)
    m2.load_state_dict(m1.state_dict())
    cfg = DPConfig(epochs=1, sample_rate=1.0, lr=1.0, momentum=0.0, max_grad_norm=0.5,
                   noise_multiplier=1e-9, seed=0)
    # neighbouring datasets: drop the last example
    train(m1, x, y, cfg)
    cfg2 = DPConfig(**{**cfg.__dict__})
    train(m2, x[:-1], y[:-1], cfg2)
    diff = torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(m1.parameters(),
                                                              m2.parameters(), strict=True)))
    # one full-batch step (q = 1, lr = 1): Δ = g_64/64 − S_63/(63·64) with every clipped
    # per-example gradient of norm <= C, so |Δ| <= C/64 + C/64 = 2C/64.
    assert float(diff) <= 2 * 0.5 / 64 + 1e-6


def test_roc_auc_matches_bruteforce():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 50), rng.normal(0, 1, 70)
    a[:5] = b[:5]  # ties
    brute = np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b])
    assert abs(roc_auc(a, b) - brute) < 1e-12
    assert 0 <= tpr_at_fpr(a, b, 0.01) <= 1


def test_eps_lower_bound_properties():
    assert eps_lower_bound(100, 50) == 0.0
    assert eps_lower_bound(100, 100) > eps_lower_bound(100, 80) > eps_lower_bound(100, 60)
    # perfect guessing on r guesses can certify at most log((1-beta)/beta)-ish growth in r
    lb = eps_lower_bound(1000, 1000, 0.05)
    p = math.exp(lb) / (1 + math.exp(lb))
    assert abs(p ** 1000 - 0.05) < 1e-3


def test_one_run_audit_detects_perfect_leak_and_not_noise():
    rng = np.random.default_rng(0)
    inc = rng.random(1000) < 0.5
    perfect = one_run_audit(inc.astype(float) + rng.normal(0, 1e-3, 1000), inc, 100, 100)
    assert perfect["accuracy"] == 1.0 and perfect["eps_lower"] > 2.5
    noise = one_run_audit(rng.normal(size=1000), inc, 100, 100)
    assert noise["eps_lower"] < 0.5
