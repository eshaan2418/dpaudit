# Data card: Fashion-MNIST

- **Source:** Xiao, Rasul and Vollgraf, *Fashion-MNIST* (2017), official idx files
  (SHA-256 checksums in `src/dpaudit/data.py`). **Licence:** MIT. Not committed.
- **Content:** 70,000 28x28 grayscale images of 10 clothing categories (60k train, 10k test).
- **Splits here:** the first 55,000 training images are the private training set, and the
  last 5,000 are a validation split used only for learning-rate selection. The 10k test set
  is used for accuracy and as the non-member population for membership inference.
- **Canaries:** 1,000 of the 55k training images, chosen per seed, get a random *wrong* label.
  Each is included in training with probability 1/2 (seeded). The 500 or so excluded
  canaries are removed from the training set. Inclusion bits are what the auditor tries to
  recover.
- **Personal data:** none (product images).
