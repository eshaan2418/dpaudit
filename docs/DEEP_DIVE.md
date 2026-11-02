# Deep dive: questions and answers

### 1. What does ε = 2 actually promise here?
For any two training sets that differ in one example, the distribution of trained models differs by at most a factor of e² ≈ 7.4 on any event, with an additive slack δ = 1e-5. That holds if the implementation matches the analysis: Poisson sampling at rate q, per-example clipping to C, Gaussian noise σC, 2,129 steps. It is a worst case over all datasets, all possible outputs, and an attacker who sees the final model (and in fact all intermediate models).

### 2. How do you know the accountant is right?
It is implemented from the formula (exact binomial expansion of the subsampled Gaussian's RDP at integer orders, then the Balle et al. 2020 conversion) and checked three ways. A Hypothesis property test against Opacus over random (q, σ, T) agrees within 1e-8 relative. The known Abadi et al. setting gives ε = 5.65. The full-batch case reduces to the Gaussian mechanism's RDP α/(2σ²). The calibrated σ also lands within 0.01 of the target ε (achieved ε 0.500, 0.999, 1.998, 3.993, 7.992).

### 3. Why Poisson sampling and division by the *expected* batch size?
The RDP bound used is for the Poisson-subsampled Gaussian mechanism. If you instead shuffle into fixed-size batches, or divide by the realised batch size, you are running a different mechanism. Dividing by the realised size makes the update's scale depend on how many examples were sampled, which is itself data-dependent. Many implementations analysed as Poisson actually use fixed batches, and that is a real accounting error. Here the code matches the analysis.

### 4. What does the sensitivity test check?
With noise turned off, it trains on two neighbouring datasets (one example removed) for one full-batch step, and checks that the parameter difference is at most 2C·lr/N. This is the exact bound for one step with mean normalisation. A bug in per-example clipping (for example clipping per parameter tensor instead of globally, or clipping after summing) would break it.

### 5. Why is the loss-threshold MIA at chance even for the non-private model?
The model is small (26k parameters) and trained for 10 epochs on 55k images. Its training and test loss distributions overlap almost completely, so a threshold on loss barely separates members from non-members (AUC 0.513). This attack measures average-case leakage, which is low for a well-generalising model. DP is a worst-case guarantee, which is why canaries (examples that are hard to fit without memorising) are the meaningful probe.

### 6. How does the one-run audit turn guesses into a lower bound?
Each canary's inclusion is an independent fair coin. Under ε-DP, conditioning on everything else, any guess about one canary's bit is correct with probability at most e^ε/(1+e^ε). Steinke et al. show this holds jointly enough that the number of correct guesses among r is dominated by Binomial(r, e^ε/(1+e^ε)). If v correct guesses would be very unlikely (p ≤ 0.05) under some ε, that ε is ruled out. The largest such ε is the lower bound. Abstaining on uncertain canaries raises accuracy on the guesses that are made, and I try 4 abstention levels with Bonferroni correction.

### 7. Why are the audited bounds ~0 when the accountant says up to 8?
Three reasons. The audit is black-box: it only sees final losses. The accountant bounds an attacker who sees every noisy gradient. The canaries are natural input-space examples whose gradients get clipped and swamped by noise, not crafted gradient canaries. And one run with 1,000 canaries gives at most ~200 confident guesses, so certifying ε = 1 needs about 73% guess accuracy, while canary AUC under DP is 0.51–0.54. The gap is expected. Published black-box one-run audits are similarly loose, while white-box audits with gradient canaries approach the theoretical bound.

### 8. Then what is the audit good for?
Detecting broken implementations, and showing the attack has power. On the non-private model the same audit certifies ε ≥ 1.43 with noise canaries (174/200 correct guesses), so the method does detect memorisation. If a DP-SGD implementation ever audited above its claimed ε, it would be proof of a bug. Here none do.

### 9. Why did noise canaries beat mislabeled canaries?
A mislabeled real image still resembles its true class, so the model's loss on the wrong label is high whether or not it was trained on it, and only partly reduced by training on it. An i.i.d. noise image is out of distribution, so nothing else in the data predicts its label, and membership determines the loss much more directly. Non-private canary AUC is 0.71 vs 0.63, and the lower bound 1.43 vs 0.50.

### 10. What is the utility cost, and how does it scale?
Accuracy goes from 89.6% non-private to 84.9% (ε = 8), 82.3% (ε = 2), and 73.6% (ε = 0.5). Gains flatten above ε ≈ 4. The cost comes from two sources: clipping (biased gradients, with 35–54% of examples clipped) and noise (σ from 0.56 to 1.84). Larger datasets help directly, because noise is added to a sum over qN examples and so its relative size shrinks as N grows.

### 11. What is not accounted for?
Hyperparameter tuning: 10 short runs on a validation split carved from the private data chose the learning rates. Rigorous practice would use public data for tuning or account for the selection (e.g. Papernot & Steinke 2022). The audit lower bounds also use the δ = 0 form of the binomial bound.

### 12. With 100x compute?
White-box audits with gradient-space (Dirac) canaries and many training runs, to see how close empirical ε gets to the accountant. Larger models and longer training, where memorisation and so audit power increase. Privacy-accounted hyperparameter selection. And a comparison of RDP against PRV/numerical accountants (Gopi et al. 2021), which give tighter ε for the same σ.
