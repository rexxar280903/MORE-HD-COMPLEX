#!/usr/bin/env python3
"""Download the torchvision MNIST files into data/MNIST/raw and verify their MD5 checksums."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.mnist import load_mnist  # noqa: E402

m = load_mnist(ROOT / "data", download=True)
print("MNIST ready:", {k: v[:16] for k, v in m.file_sha256.items()})
