# Bare-Metal Inference Engine

A C++ neural network inference engine built from scratch — no frameworks, no dependencies, no libraries. Just raw matrix operations, manual memory management, int8 quantization, AVX2 SIMD acceleration, CUDA GPU kernels, and a Transformer with KV caching.

**Built to understand how AI inference actually works at the systems level.**

**Machine:** Intel i5-13th Gen (CPU) · NVIDIA T4 (GPU)
**Compiler:** g++ `-O3 -march=native -mavx2` / nvcc `-O2 -arch=sm_75`

---

## Why This Project

Low-latency inference is the backbone of modern AI systems — from real-time trading signals (HFT) to large language model serving. This project builds every layer of the inference stack from raw C++, exposing the tradeoffs between precision, quantization, SIMD vectorization, GPU acceleration, and autoregressive generation.

If you're hiring for **AI Infrastructure** or **Systems Engineering** roles, this codebase demonstrates that I don't just *use* frameworks — I understand the hardware constraints inside them.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                            │
│                                                                 │
│  Month 1          Month 2              Month 3                   │
│  ────────         ────────             ────────                  │
│  Float32     →    Int8 Quant     →     CUDA Naive                │
│  CPU Baseline      CPU Scalar          GPU Kernel                │
│       │                │                     │                   │
│       ▼                ▼                     ▼                   │
│  Forward Pass    AVX2 SIMD          Shared-Memory Tiling         │
│  Binary I/O      SIMD Int8          Batched Throughput           │
│       │                │             Scaled Stress Tests         │
│       ▼                ▼                     │                   │
│  Tensor+Layer   3-Way Benchmark            ▼                    │
│  ReLU Activ.    Latency Histogram    Transformer + KV Cache      │
│                                     Attention → LayerNorm → MLP  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
├── src/                            Month 1 — float32 foundations
│   ├── tensor.hpp / tensor.cpp       Tensor struct + matmul()
│   ├── Layer.hpp                     Fully connected layer
│   ├── relu.hpp                      ReLU activation
│   ├── main.cpp                      Week 1: matmul baseline
│   ├── mainv2.cpp                    Week 2: 3-layer forward pass
│   └── inference.cpp                 Week 4: binary weight loader
│
├── v1_float32/                     Month 2 — float32 baseline
│   └── main.cpp                       Self-contained float32 inference
│
├── v2_int8/                        Month 2 — int8 quantized engine
│   ├── main.cpp                       Self-contained int8 inference
│   └── weights_int8.bin               Quantized weights (int8 + scale)
│
├── v3_simd/                        Month 2 — AVX2 SIMD int8 engine
│   ├── int8_simd.hpp                  Scalar + SIMD matmul implementations
│   ├── main.cpp                       Standalone SIMD inference
│   ├── validate.cpp                   Correctness validation (10 inputs)
│   └── bench_simd.cpp                 Triple cross-check benchmark
│
├── v4_cuda/                        Month 2-3 — CUDA GPU engine
│   ├── cuda_inference.cu              Naive CUDA kernel (Month 2)
│   ├── cuda_shared.cu                 16×16 shared-memory tiled matmul
│   ├── cuda_pipeline.cu               Full CPU→GPU→CPU pipeline
│   ├── cuda_batched.cu                Batched kernel + throughput sweep
│   └── profile_kernel.cu              NSight Compute profiling harness
│
├── v5_transformer/                 Month 3 — Transformer + KV Cache
│   ├── attention.hpp                  Self-attention (Q/K/V projections)
│   ├── transformer_block.hpp          Attention + LayerNorm + MLP
│   ├── kv_cache.hpp                   KV cache for autoregressive generation
│   └── main_transformer.cpp           Full Transformer benchmark driver
│
├── v6_scaled/                      Month 3 — Scaled Stress Tests
│   ├── bench_scaled.cpp               CPU scaled matmul + KV cache stress
│   └── bench_scaled_cuda.cu           GPU naive vs shared memory at scale
│
├── benchmarks/
│   ├── bench.cpp                      128×128 matmul latency
│   ├── bench_both.cpp                 Month 2: float32 vs int8 benchmark
│   ├── bench_simd.cpp                 Month 2: scalar vs SIMD benchmark
│   ├── cache_experiment.cpp           Cache line experiment
│   ├── alignment_demo.cpp             Memory alignment check
│   ├── latency_results.csv            float32 vs int8 raw data (10K samples)
│   ├── latency_results_v3.csv         3-way scalar vs SIMD raw data
│   ├── latency_results_shared.csv     Naive vs shared-memory CUDA
│   ├── latency_results_pipeline.csv   Pipeline component breakdown
│   ├── throughput_results.csv         Batched throughput sweep
│   ├── scaled_results.txt             Scaled CPU raw data
│   ├── scaled_results_gpu.txt         Scaled GPU raw data
│   ├── latency_histogram.png          Month 2 latency chart
│   ├── latency_histogram_v2.png       Month 2 3-way chart
│   └── latency_histogram_final.png    Month 3 5-config comparison
│
├── scripts/
│   ├── generate_weights.py            Weight generator (numpy, seed=42)
│   ├── golden.py                      Golden reference output
│   ├── quantize_weights.py            int8 quantization script
│   ├── plot_histogram.py              Month 2 histogram plotter
│├── plot_histogram_v3.py           Month 2 3-way histogram plotter
│   └── plot_final_histogram.py        Month 3 5-config aggregator
│
├── weights.bin                      Pretrained weights (float32 binary)
└── Makefile
```

---

## Build & Run

```bash
make all            # Build all targets
make weights        # Regenerate weights.bin
make debug          # Build with -O0 for debugging
make clean          # Remove binaries

# Month 1 & 2 (CPU)
./engine            # Week 1: matmul baseline
./float32_v1        # Float32 baseline
./int8              # Int8 quantized inference

# Month 3 (Transformer & CUDA)
g++ -O3 -march=native -mavx2 -std=c++17 -Isrc -o transformer v5_transformer/main_transformer.cpp src/tensor.cpp
./transformer

nvcc -O2 -arch=sm_75 -o cuda_shared v4_cuda/cuda_shared.cu
./cuda_shared v2_int8/weights_int8.bin

# Month 3 (Scaled Stress Tests)
g++ -O3 -march=native -mavx2 -std=c++17 v6_scaled/bench_scaled.cpp -o bench_scaled
./bench_scaled

nvcc -O2 -arch=sm_75 -std=c++17 v6_scaled/bench_scaled_cuda.cu -o bench_scaled_cuda
./bench_scaled_cuda
```

---

## Month 1 Progress

| Week | Focus | Deliverable |
|------|-------|-------------|
| **Week 1** | C++ foundations | `Tensor` struct, `matmul()`, chrono benchmarking |
| **Week 2** | Neural net forward pass | `Layer` struct, 3-layer ReLU network, end-to-end timing |
| **Week 3** | Memory & cache model | Cache line experiment, alignment demo |
| **Week 4** | Binary I/O | Python weight generator, `./inference`, golden validation |

### Month 1 Benchmarks

| Metric | Value |
|--------|-------|
| 128×128 matmul (p50) | 274 µs |
| 3-layer forward pass (p50) | ~1 µs |
| Sequential memory (1M floats) | 733 µs |
| Strided memory (every 16th) | 5502 µs |
| Cache penalty | 7.5× |
| Output error vs golden | 4.46e-07 |

---

## Month 2 Progress

| Week | Focus | Deliverable |
|------|-------|-------------|
| **Week 5** | int8 quantization | `quantize_weights.py`, `weights_int8.bin`, scale-factor check |
| **Week 6** | int8 forward pass + benchmarking | `v2_int8/` inference, 10K-sample p50/p95/p99, latency histogram |
| **Week 7** | AVX2 SIMD int8 matmul | `v3_simd/` engine, 10-input validation, triple cross-check benchmark, 3-way histogram |
| **Week 8** | CUDA GPU Kernel | `v4_cuda/` naive CUDA implementation, custom block reduction quantization kernel, correctness validation |

### Month 2 Benchmarks

**Latency (batch=1 single-call, 1K samples):**

| Metric | float32 | int8 scalar | int8 SIMD (AVX2) | SIMD vs scalar |
|--------|---------|-------------|-------------------|----------------|
| p50 | 0.801 µs | 2.071 µs | 2.052 µs | 1.01× |
| p95 | 1.129 µs | 2.903 µs | 2.088 µs | 1.4× |
| p99 | 1.271 µs | 2.974 µs | 2.124 µs | 1.4× |

> Note: SIMD provides no meaningful speedup at this model size (1.01×). The network is tiny (max 128×64 matmul). The overhead of loading data into SIMD registers, widening int8→int16→int32, and storing results back dominates the actual arithmetic savings.

---

## Month 3 Progress

| Week | Focus | Deliverable |
|------|-------|-------------|
| **Week 9** | CUDA Shared Memory & Loop Unrolling | `cuda_shared.cu` with 16×16 tiled matmul, `#pragma unroll` for register blocking |
| **Week 10** | Transformer + KV Cache | `v5_transformer/` — Self-attention, LayerNorm, MLP, autoregressive generation with KV caching |
| **Week 11** | Scaling Validation | `v6_scaled/` stress tests proving crossover points for SIMD, Int8, and Shared Memory at 512×512 and 1024×1024 |

### Month 3 Benchmarks

#### 1. Transformer + KV Cache (Memory Access Bug Fix)

During initial testing, the KV cache implementation suffered a 47× slowdown compared to full recomputation. Profiling revealed a memory access pattern bug in the attention loop.

**Latency (seq_len=50, single-token generation):**

| Configuration | Latency (µs) | Speedup |
|---------------|-------------|---------|
| No cache (full recomputation) | 598.984 | 1.0× |
| With KV cache (pre-allocated, zero-alloc) | 14.384 | **41.642×** |

**Root Cause & Fix:**
The original inner loop accessed `V_cache[p][d]` with `p` in the inner loop, causing stride-`d_head` memory access and catastrophic L1/L2 cache misses. Swapping the loop order to `p { d { } }` allowed sequential memory access, turning a 47× slowdown into a 41.6× speedup.

#### 2. CUDA Shared Memory (Small Scale)

| Kernel | p50 (µs) | Speedup |
|--------|----------|---------|
| Naive CUDA | 42.88 | 1.0× |
| Shared Memory Tiled + `#pragma unroll` | 38.53 | 1.113× |

> Note: Shared memory tiling provides minimal speedup at this model size. The matrix dimensions are too small for the tiling overhead to pay off.

#### 3. CPU vs GPU Crossover

| Batch Size | CPU (SIMD) per query | GPU total (µs) | GPU QPS | Winner |
|------------|----------------------|----------------|---------|--------|
| 1 | 2.05 µs | 39.26 | 25,469 | CPU |
| 8 | 16.36 µs | 29.25 | 273,523 | CPU |
| 16 | 32.72 µs | 31.46 | 508,647 | **Crossover** |
| 32 | 65.44 µs | 36.29 | 881,834 | GPU |

GPU throughput overtakes CPU at **batch_size = 16**. PCIe transfer overhead (~25 µs) is amortized across all queries in a batch.

#### 4. Scaling Validation: CPU Int8 Quantization (512×512 & 1024×1024)

To prove the optimizations scale beyond the tiny 128×64 baseline, the engine was stress-tested at larger matrix sizes.

**Latency (512×512 Matmul):**

| Implementation | p50 (µs) | p95 (µs) | p99 (µs) | Int8 vs Float32 |
|----------------|----------|----------|----------|-----------------|
| Float32 Scalar | 13,873.77 | 22,426.46 | 24,682.32 | 1.0× |
| Int8 Scalar | 13,310.89 | 18,134.39 | 19,325.97 | 1.04× |
| Int8 AVX2 SIMD | 14,585.47 | 20,594.88 | 22,597.72 | 0.95× |

**Latency (1024×1024 Matmul):**

| Implementation | p50 (µs) | p95 (µs) | p99 (µs) | Int8 vs Float32 |
|----------------|----------|----------|----------|-----------------|
| Float32 Scalar | 146,213.11 | 205,176.78 | 220,268.78 | 1.0× |
| Int8 Scalar | 106,464.50 | 141,248.60 | 148,227.75 | **1.37×** |
| Int8 AVX2 SIMD | 109,912.44 | 159,128.38 | 173,986.45 | 1.33× |

**Analysis:** At 128×64, Int8 quantization was 1.8× slower than Float32 due to overhead. At 1024×1024, Int8 scalar is **1.37× faster**. However, Int8 scalar beats Int8 AVX2 SIMD because AVX2 lacks a native Int8 multiply-accumulate (MAC) instruction, requiring expensive Int8→Int16→Int32 widening. Modern Intel CPUs feature hardware FMA for Float32, allowing the compiler's auto-vectorizer to execute scalar loops extremely efficiently.

#### 5. Scaling Validation: GPU Shared Memory (512×512 & 1024×1024)

Shared memory tiling underperformed at 128×64 (1.11×) due to overhead. At larger sizes, it provides a massive speedup.

| Matrix Size | Kernel | p50 (µs) | p95 (µs) | p99 (µs) | Speedup |
|-------------|--------|----------|----------|----------|---------|
| 512×512 | Naive Global | 490.46 | 852.13 | 857.31 | 1.0× |
| 512×512 | Shared Memory | 306.82 | 311.39 | 316.80 | **1.599×** |
| 1024×1024 | Naive Global | 4,368.67 | 4,426.69 | 4,486.88 | 1.0× |
| 1024×1024 | Shared Memory | 2,650.11 | 2,697.82 | 2,700.77 | **1.648×** |

#### 6. Scaling Validation: Transformer KV Cache (seq_len=256)

The KV cache scales perfectly from O(n²) to O(n), reducing 256-token generation time from 1.3 seconds to 6 milliseconds.

| Configuration (256 tokens) | Total Time | Speedup |
|---------------------------|------------|---------|
| No Cache (Full Recomputation) | 1,334,731 µs (1.3 s) | 1.0× |
| With KV Cache | 6,153 µs (0.0 s) | **216.9×** |

---

## Roadmap

- [x] **Month 1** — C++ foundations, forward pass, binary weights
- [x] **Month 2** — int8 quantization, AVX2 SIMD, CUDA GPU kernel
- [x] **Month 3** — Shared-memory tiling, batched throughput, Transformer + KV cache, scaling validation

---

## License

MIT
