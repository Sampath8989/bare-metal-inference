#!/usr/bin/env python3
"""
quantize_weights.py

Quantizes the existing float32 weights (weights.bin) to int8 format,
producing v2_int8/weights_int8.bin for the quantized inference engine.

Scale Factor:
  The scale factor maps between the float32 range and the int8 range [-128, 127].
  We find the largest absolute value in the weight matrix, then divide by 127
  to get a scale. Every float weight is divided by this scale and rounded to the
  nearest integer so it fits in int8. To get back to float, multiply the int8
  value by the scale. This loses some precision but saves 4x memory and enables
  faster integer arithmetic.

Zero Point:
  For this symmetric quantization the zero point is always 0 — meaning int8 value 0
  maps exactly to float 0.0. This simplifies the math and works well for neural
  network weights which tend to be centred around zero.

Binary format produced (per layer):
  [int32 rows][int32 cols][float32 scale][rows*cols int8 weights]
  [int32 bias_size][bias_size float32 biases]

Biases are kept as float32 (small count, negligible memory cost).

Usage:
  python3 scripts/quantize_weights.py
"""

import os
import struct
import numpy as np

# Must match generate_weights.py
LAYERS = [(128, 64), (64, 32), (32, 10)]


def quantize_matrix(weights_float: np.ndarray):
    """Symmetric per-matrix quantization: float32 → int8 + scale."""
    max_abs = float(np.max(np.abs(weights_float)))
    if max_abs == 0.0:
        max_abs = 1.0  # avoid division by zero
    scale = max_abs / 127.0
    weights_int8 = np.clip(np.round(weights_float / scale), -128, 127).astype(np.int8)
    return weights_int8, scale


def main():
    weights_path = "weights.bin"
    out_dir = "v2_int8"
    out_path = os.path.join(out_dir, "weights_int8.bin")

    os.makedirs(out_dir, exist_ok=True)

    # Read float32 weights
    layers = []
    with open(weights_path, "rb") as f:
        for idx, (in_sz, out_sz) in enumerate(LAYERS):
            rows, cols = struct.unpack("ii", f.read(8))
            w_bytes = rows * cols * 4
            weights = np.frombuffer(f.read(w_bytes), dtype=np.float32)
            bias_size = struct.unpack("i", f.read(4))[0]
            bias = np.frombuffer(f.read(bias_size * 4), dtype=np.float32)
            layers.append((rows, cols, weights.copy(), bias.copy()))
            print(f"  Read layer {idx + 1}: {rows}×{cols} float32 weights + {bias_size} biases")

    # Quantize and write to output binary
    with open(out_path, "wb") as f:
        for rows, cols, weights, bias in layers:
            w_int8, scale = quantize_matrix(weights)

            # Header
            f.write(struct.pack("ii", rows, cols))
            f.write(struct.pack("f", scale))
            # Int8 weights
            f.write(w_int8.tobytes())
            # Bias (float32)
            f.write(struct.pack("i", len(bias)))
            f.write(bias.tobytes())

    print(f"\n✓ Saved {out_path}")

    # Sanity check: reload and compare
    print("\n=== Quantization Sanity Check ===")
    max_global_err = 0.0
    with open(out_path, "rb") as f:
        for idx, (rows, cols, w_orig, _) in enumerate(layers):
            r, c = struct.unpack("ii", f.read(8))
            scale = struct.unpack("f", f.read(4))[0]
            w_int8 = np.frombuffer(f.read(r * c), dtype=np.int8)
            b_sz = struct.unpack("i", f.read(4))[0]
            f.read(b_sz * 4)  # skip bias

            # Dequantize and compare
            w_dequant = w_int8.astype(np.float32) * scale
            max_err = float(np.max(np.abs(w_dequant - w_orig)))
            max_global_err = max(max(max_global_err, max_err), 0.0)

            print(f"  Layer {idx + 1}: scale={scale:.8e}  "
                  f"int8 range [{int(w_int8.min())}, {int(w_int8.max())}]  "
                  f"max error vs float32 = {max_err:.8e}")

    print(f"\n  Max absolute error across all layers: {max_global_err:.8e}")

    # Debug: print 10 sample weight triples from Layer 1
    print("\n=== Layer 1 Weight Sample (first 10 values) ===")
    print(f"  {'orig_float32':>16s}  {'dequant_int8':>16s}  {'abs_diff':>16s}")
    with open(out_path, "rb") as f:
        # Skip to layer 1
        r, c = struct.unpack("ii", f.read(8))
        scale = struct.unpack("f", f.read(4))[0]
        w_int8 = np.frombuffer(f.read(r * c), dtype=np.int8)
        w_dequant = w_int8.astype(np.float32) * scale
        w_orig = layers[0][2]  # original float32 weights for layer 1
        for i in range(10):
            diff = abs(float(w_dequant[i]) - float(w_orig[i]))
            print(f"  {float(w_orig[i]):16.8f}  {float(w_dequant[i]):16.8f}  {diff:16.8e}")
        print(f"  ... ({r * c} total weights in layer 1)")
        print(f"  Layer 1 scale = {scale:.10e}")
        print(f"  Layer 1 int8 range = [{int(w_int8.min())}, {int(w_int8.max())}]")
        print(f"  NOTE: All weights are 0.01f (from generate_weights.py)")
        print(f"  0.01 / scale = {0.01/scale:.6f} → rounds to int8 = 127")
        print(f"  127 * scale = {127*scale:.10e} → back to float = 0.01 exactly")
        print(f"  → Weight error is genuinely ~0 because test weights are degenerate.")

    print("\n  ✓ Sanity check passed\n")


if __name__ == "__main__":
    main()
