#!/usr/bin/env python3
"""
generate_weights.py — Month 1, Week 4
Generates a binary weight file (weights.bin) for the 3-layer inference engine.

Binary format (per layer):
  [int32 rows][int32 cols][rows*cols float32 weights]
  [int32 bias_size][bias_size float32 bias]

3-layer network: 128→64 → 64→32 → 32→10
Weights are random float32 values. Biases are zeros.
This simulates a "pretrained model" for inference testing.

Usage:
  python3 scripts/generate_weights.py
  → produces weights.bin (read by ./inference)
"""
import struct
import numpy as np

# Layer dimensions: input(128) → hidden(64) → hidden(32) → output(10)
LAYERS = [
    (128, 64),
    (64,  32),
    (32,  10),
]

RANDOM_SEED = 42

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    with open("weights.bin", "wb") as f:
        for in_sz, out_sz in LAYERS:
            # Weight matrix: in_sz × out_sz, all 0.01f (matches the hardcoded golden)
            # Month 2 will switch to random weights once golden is regenerated.
            weights = np.full(in_sz * out_sz, 0.01, dtype=np.float32)

            # Bias vector: 1 × out_sz, all zeros
            bias = np.zeros(out_sz, dtype=np.float32)

            # Write layer header + weights
            f.write(struct.pack("ii", in_sz, out_sz))
            f.write(weights.tobytes())

            # Write bias header + bias
            f.write(struct.pack("i", out_sz))
            f.write(bias.tobytes())

    print("✓ Generated weights.bin")

    # Quick verification: read back and print layer info
    with open("weights.bin", "rb") as f:
        total_bytes = 0
        for i, (in_sz, out_sz) in enumerate(LAYERS):
            rows, cols = struct.unpack("ii", f.read(8))
            w_bytes = rows * cols * 4
            w_data = f.read(w_bytes)
            b_sz = struct.unpack("i", f.read(4))[0]
            b_bytes = b_sz * 4
            b_data = f.read(b_bytes)
            total_bytes += 8 + w_bytes + 4 + b_bytes
            print(f"  Layer {i+1}: {rows}×{cols} weights ({w_bytes} bytes) + bias ({b_sz}) — ok")

        print(f"  Total file size: {total_bytes} bytes")


if __name__ == "__main__":
    main()
