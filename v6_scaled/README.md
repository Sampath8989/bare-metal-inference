# V6 Scaled Inference Engine Benchmarks

V6 is the scaled benchmarking module for comparing neural network inference engines across matrix and model dimensions of **512×512** and **1024×1024**.

---

## Architecture Overview

- **V1 (Float32)**: Baseline 3-layer MLP inference using single-precision floating point.
- **V2 (Int8)**: Quantized Int8 3-layer MLP with Int32 accumulation and Float32 dequantization + ReLU.
- **V3 (AVX2 SIMD)**: Int8 inference accelerated with x86 AVX2 256-bit SIMD vector instructions (processing 32 int8 values per iteration).
- **V4 (CUDA)**: GPU-accelerated inference using CUDA kernels (naive and 16×16 shared-memory tiled).
- **V5 (Transformer)**: Single-head self-attention + Transformer Block comparing **No-Cache** vs **KV Cache** generation.
- **V6 (Scaled Benchmark)**: Top-level unified suite to build and evaluate V1–V5 at scaled dimensions (512 and 1024).

---

## Model & Matrix Configurations

| Version | Matrix / Model Dimensions (512 Config) | Matrix / Model Dimensions (1024 Config) |
|---|---|---|
| **V1 Float32** | 512 → 512 → 512 → 512 | 1024 → 1024 → 1024 → 1024 |
| **V2 Int8** | 512 → 512 → 512 → 512 | 1024 → 1024 → 1024 → 1024 |
| **V3 SIMD** | 512 → 512 → 512 → 512 | 1024 → 1024 → 1024 → 1024 |
| **V4 CUDA** | 512 → 512 → 512 → 512 | 1024 → 1024 → 1024 → 1024 |
| **V5 Transformer** | `d_model`=512, `d_head`=64, `mlp`=2048 | `d_model`=1024, `d_head`=128, `mlp`=4096 |

---

## Current Measured Results

### 1. 3-Layer MLP Scaling (p50 Latency)

| Dimension | V1 Float32 Baseline | V2 Int8 Quantized | V3 Int8 AVX2 SIMD | Speedup vs Baseline (V3) |
|---|---|---|---|---|
| **512×512** | 66.00 µs | 62.58 µs | 59.98 µs | **1.10×** |
| **1024×1024** | 396.00 µs | 244.07 µs | 237.85 µs | **1.66×** |

### 2. Transformer Block + KV Cache Autoregressive Generation (seq_len=50)

| Dimension | No Cache (Full Recomputation) | With KV Cache (Pre-allocated) | KV Cache Speedup |
|---|---|---|---|
| **512 Config** (`d_model`=512, `mlp`=2048) | 8,006.83 µs | 200.23 µs | **39.99×** |
| **1024 Config** (`d_model`=1024, `mlp`=4096) | 82,810.35 µs | 1,459.63 µs | **56.73×** |

### 3. GPU CUDA Stress Test (NVIDIA Tesla T4)

| Matrix Size | Naive Global CUDA (p50) | Shared-Memory Tiled CUDA (p50) | Speedup |
|---|---|---|---|
| **512×512** | 490.46 µs | 306.82 µs | **1.599×** |
| **1024×1024** | 4,368.67 µs | 2,650.11 µs | **1.648×** |

---

## How to Build & Run

### Build Everything & Run 512 Configuration
```bash
make v6_scaled_512
```

### Build Everything & Run 1024 Configuration
```bash
make v6_scaled_1024
```

### Run Individual Components
```bash
make -C v6_scaled
./v6_scaled/v1_float32_exe 512
./v6_scaled/v2_int8_exe 512
./v6_scaled/v3_simd_exe 512
./v6_scaled/v5_transformer_exe 512
```
