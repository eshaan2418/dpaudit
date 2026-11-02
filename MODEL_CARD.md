# Model card

- **Models:** a tanh CNN with 26,010 parameters (two conv layers, two linear layers), trained
  with DP-SGD at ε ∈ {0.5, 1, 2, 4, 8} (δ = 1e-5) and non-privately, for 10 epochs at an
  expected batch of 256. Weights are not committed.
- **Intended use:** studying privacy-utility trade-offs and empirically auditing DP-SGD
  implementations and accountants.
- **Not intended for:** treating the audit's lower bound as a privacy guarantee. Only the
  accountant's upper bound is a guarantee, and only if the implementation is correct and
  the hyperparameter search is accounted for (it is not here; see README).
- **Limitations:** a black-box audit with input-space canaries gives loose lower bounds.
  White-box gradient canaries would give tighter ones.
