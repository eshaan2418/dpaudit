"""dpaudit CLI.

    dpaudit epsilon --q 0.0047 --sigma 0.87 --steps 2129      # accountant
    dpaudit sigma --eps 2 --q 0.0047 --steps 2129              # calibrate noise
    dpaudit lower-bound --guesses 200 --correct 140            # audit bound from guesses
"""

from __future__ import annotations

import argparse

from .accountant import calibrate_sigma, epsilon
from .audit import eps_lower_bound


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dpaudit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("epsilon")
    s = sub.add_parser("sigma")
    for p in (e, s):
        p.add_argument("--q", type=float, required=True)
        p.add_argument("--steps", type=int, required=True)
        p.add_argument("--delta", type=float, default=1e-5)
    e.add_argument("--sigma", type=float, required=True)
    s.add_argument("--eps", type=float, required=True)
    lb = sub.add_parser("lower-bound")
    lb.add_argument("--guesses", type=int, required=True)
    lb.add_argument("--correct", type=int, required=True)
    lb.add_argument("--beta", type=float, default=0.05)
    a = ap.parse_args(argv)
    if a.cmd == "epsilon":
        eps, order = epsilon(a.q, a.sigma, a.steps, a.delta)
        print(f"epsilon = {eps:.4f} at delta = {a.delta:g} (best RDP order {order})")
    elif a.cmd == "sigma":
        sig = calibrate_sigma(a.eps, a.q, a.steps, a.delta)
        print(f"noise multiplier sigma = {sig:.4f} for epsilon <= {a.eps:g}")
    else:
        v = eps_lower_bound(a.guesses, a.correct, a.beta)
        print(f"{a.correct}/{a.guesses} correct -> epsilon >= {v:.3f} "
              f"with confidence {1 - a.beta:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
