"""One DP-SGD run on FashionMNIST with canaries: utility, membership inference, one-run audit.

    python scripts/experiment.py --eps 2 --seed 0 [--lr 2.0] [--epochs 10] [--tune]

Data: the 60k training images are split into 55k train / 5k validation (fixed). m=1000 canaries
(drawn from the 55k, given a random *wrong* label) are each included with probability 1/2;
excluded canaries are removed from the training set, so the inclusion bit is the secret the
auditor tries to recover. σ is calibrated so that the full run satisfies (ε, δ=1e-5)-DP under
the RDP accountant. --tune evaluates on the validation split instead of the test split.
Writes eval/results/{runs,tune}/eps<ε>_s<seed>[_lr<lr>].json.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from dpaudit.accountant import calibrate_sigma, epsilon
from dpaudit.audit import best_one_run_audit, roc_auc, tpr_at_fpr
from dpaudit.data import fashion_mnist
from dpaudit.dpsgd import DPConfig, SmallCNN, accuracy, per_example_loss, train

DELTA = 1e-5
M_CANARIES = 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, required=True, help="target epsilon; inf = non-private")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--canary", choices=["mislabeled", "noise"], default="mislabeled",
                    help="mislabeled: real images with a wrong label; noise: i.i.d. Gaussian "
                         "images with a random label (out-of-distribution, easier to audit)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    xtr, ytr, xte, yte = fashion_mnist()
    xval, yval = xtr[55000:], ytr[55000:]
    xtr, ytr = xtr[:55000], ytr[:55000]
    rng = np.random.default_rng(1000 + a.seed)
    canary_idx = rng.choice(55000, M_CANARIES, replace=False)
    included = rng.random(M_CANARIES) < 0.5
    y_can = ytr.clone()
    wrong = (ytr[canary_idx].numpy() + rng.integers(1, 10, M_CANARIES)) % 10
    y_can[canary_idx] = torch.from_numpy(wrong)
    if a.canary == "noise":
        xtr = xtr.clone()
        gen = torch.Generator().manual_seed(2000 + a.seed)
        xtr[canary_idx] = torch.randn(M_CANARIES, 1, 28, 28, generator=gen) * 2.0
    keep = np.ones(55000, dtype=bool)
    keep[canary_idx[~included]] = False
    x_train, y_train = xtr[keep], y_can[keep]
    n = len(x_train)
    q = 256 / n
    steps = int(round(a.epochs / q))
    private = math.isfinite(a.eps)
    sigma = calibrate_sigma(a.eps, q, steps, DELTA) if private else None
    # learning rates chosen on the validation split (scripts/tune.sh, eval/results/tune)
    lr = a.lr if a.lr is not None else (0.25 if private else 0.1)
    cfg = DPConfig(epochs=a.epochs, sample_rate=q, lr=lr, noise_multiplier=sigma, seed=a.seed)
    torch.manual_seed(a.seed)
    model = SmallCNN()
    stats = train(model, x_train, y_train, cfg)
    x_eval, y_eval = (xval, yval) if a.tune else (xte, yte)
    rec = {"target_eps": a.eps, "seed": a.seed, "lr": lr, "canary": a.canary,
           "epochs": a.epochs, "q": q,
           "steps": steps, "sigma": sigma, "delta": DELTA,
           "achieved_eps": epsilon(q, sigma, steps, DELTA)[0] if private else None,
           "n_train": n, "accuracy": accuracy(model, x_eval, y_eval),
           "eval_split": "val" if a.tune else "test", **stats, "git_commit": commit}
    if not a.tune:
        # membership inference: 10k random (non-canary) training members vs the 10k test set
        non_canary = np.setdiff1d(np.arange(55000), canary_idx)
        mem = rng.choice(non_canary, 10000, replace=False)
        li = per_example_loss(model, xtr[mem], ytr[mem])
        lo = per_example_loss(model, xte, yte)
        rec["mia_auc"] = roc_auc(-li, -lo)
        rec["mia_tpr_at_1pct_fpr"] = tpr_at_fpr(-li, -lo, 0.01)
        rec["mia_tpr_at_0.1pct_fpr"] = tpr_at_fpr(-li, -lo, 0.001)
        # one-run audit on canaries (score = -loss on the canary's wrong label)
        sc = -per_example_loss(model, xtr[canary_idx], y_can[canary_idx])
        rec["canary_auc"] = roc_auc(sc[included], sc[~included])
        rec["audit"] = best_one_run_audit(sc, included, ks=(25, 50, 100, 200))
    out = Path("eval/results") / ("tune" if a.tune else
                                  ("runs" if a.canary == "mislabeled" else "runs_noise"))
    out.mkdir(parents=True, exist_ok=True)
    name = f"eps{a.eps}_s{a.seed}" + (f"_lr{lr}" if a.tune else "")
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in rec if k in ("target_eps", "seed", "lr", "sigma",
                                                     "accuracy", "mia_auc", "canary_auc")}
                     | ({"eps_lower": rec["audit"]["eps_lower"]} if "audit" in rec else {})),
          flush=True)


if __name__ == "__main__":
    main()
