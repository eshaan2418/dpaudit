"""FashionMNIST (MIT licence) from the official idx files, SHA-256 checked."""

from __future__ import annotations

import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import torch

BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FILES = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}
MEAN, STD = 0.2860, 0.3530  # FashionMNIST training-set pixel statistics


def _read(path: Path, offset: int) -> np.ndarray:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FILES[path.name]:
        raise SystemExit(f"{path.name}: checksum mismatch")
    return np.frombuffer(gzip.decompress(raw), dtype=np.uint8, offset=offset)


def fashion_mnist(cache: str | Path = "data") -> tuple[torch.Tensor, ...]:
    d = Path(cache)
    d.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (d / f).exists():
            urllib.request.urlretrieve(BASE + f, d / f)
    xtr = _read(d / "train-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28)
    ytr = _read(d / "train-labels-idx1-ubyte.gz", 8)
    xte = _read(d / "t10k-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28)
    yte = _read(d / "t10k-labels-idx1-ubyte.gz", 8)
    f = lambda a: (torch.from_numpy(a.astype(np.float32)) / 255 - MEAN) / STD
    return f(xtr), torch.from_numpy(ytr.astype(np.int64)), f(xte), torch.from_numpy(
        yte.astype(np.int64))
