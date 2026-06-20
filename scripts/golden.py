#!/usr/bin/env python3
"""
golden.py
Generates golden reference output for the 3-layer inference engine.

Loads weights dynamically from weights.bin rather than hardcoding.
Uses ReLU on all 3 layers to match the baseline float32 C++ model architecture.

Usage:
  python3 scripts/golden.py
  → produces golden_output.txt
"""
import struct
import numpy as np

LAYERS = [(128, 64), (64, 32), (32, 10)]

def main():
    weights_path = "weights.bin"
    layers = []
    with open(weights_path, "rb") as f:
        for idx, (in_sz, out_sz) in enumerate(LAYERS):
            rows, cols = struct.unpack("ii", f.read(8))
            w_bytes = rows * cols * 4
            weights = np.frombuffer(f.read(w_bytes), dtype=np.float32).reshape(rows, cols)
            bias_size = struct.unpack("i", f.read(4))[0]
            bias = np.frombuffer(f.read(bias_size * 4), dtype=np.float32)
            layers.append((weights, bias))

    # Input vector: all 1.0f
    x = np.ones((1, 128), dtype=np.float32)

    # Layer 0 (128 -> 64) with ReLU
    o1 = np.maximum(0, x @ layers[0][0] + layers[0][1])
    # Layer 1 (64 -> 32) with ReLU
    o2 = np.maximum(0, o1 @ layers[1][0] + layers[1][1])
    # Layer 2 (32 -> 10) with ReLU (matching C++ baseline default Layer::forward behaviour)
    o3 = np.maximum(0, o2 @ layers[2][0] + layers[2][1])

    print("=== GOLDEN REFERENCE OUTPUT ===")
    for i, v in enumerate(o3[0]):
        print(f"  golden[{i}] = {v:.6f}")

    np.savetxt("golden_output.txt", o3, fmt="%.6f")
    print("\nSaved to golden_output.txt")

if __name__ == "__main__":
    main()
