# Design

## DP-SGD
- **Per-example gradients** with `torch.func.vmap(grad(loss))` over a functional model call.
  There are no hooks and no Opacus at training time, so the clipping and noise code is
  short enough to read in full (`dpsgd.py`).
- **Poisson sampling:** every step includes each example independently with probability q.
  The noisy sum is divided by the *expected* batch size qN, not the realised one. Dividing by
  the realised size would make the normalisation depend on the data and break the analysis
  the accountant assumes.
- **Clipping** is over the concatenation of all parameter gradients (global L2 norm per
  example), and noise `N(0, σ²C²)` is added per coordinate to the clipped sum.
- **Momentum SGD with cosine LR decay.** Post-processing of the noisy gradients does not
  change the privacy guarantee.

## Accountant
The RDP of the Poisson-subsampled Gaussian at integer orders 2..256 uses the exact binomial
expansion (Mironov et al. 2019), composed over T steps and converted to (ε, δ) with the
improved bound of Balle et al. (2020). Property tests compare it against Opacus over random
(q, σ, T), and the agreement is within 1e-8 relative. σ is calibrated per target ε by
bisection.

## Audits
- **Membership inference:** a loss-threshold attack on 10k random training members vs the 10k
  test images. This is the standard weak baseline. It measures average-case leakage, not
  the worst case DP bounds.
- **One-run audit** (Steinke, Nasr, Jagielski 2023): the inclusion bits of 1,000 mislabeled
  canaries are guessed from their training loss. The auditor guesses IN for the k
  lowest-loss and OUT for the k highest-loss canaries, abstains on the rest, and converts
  the number of correct guesses into a lower bound on ε with a binomial tail at 95%
  confidence (Bonferroni-corrected over 4 values of k). It needs a single training run.
