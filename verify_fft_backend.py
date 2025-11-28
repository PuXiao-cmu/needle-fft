#!/usr/bin/env python3
"""
Quick test to verify FFT backend is being called correctly.
Usage: NEEDLE_FFT_IMPL=cpp python verify_fft_backend.py
"""

import sys
import os

# Print environment variable BEFORE importing
print(f"Environment variable NEEDLE_FFT_IMPL = {os.environ.get('NEEDLE_FFT_IMPL', 'not set')}")

sys.path.insert(0, './python')

import numpy as np
import needle
from needle import Tensor
import needle.ops as ops

# Create a small tensor
device = needle.cpu_numpy()
x = Tensor(np.random.randn(2, 28).astype(np.float32), device=device)

print(f"\nCalling FFT on tensor with shape {x.shape}...")
result = ops.fft(x, dim=-1, norm="backward")

print(f"FFT completed successfully!")
print(f"Result shape: {result[0].shape}")
