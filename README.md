# low-latency-inference

A C++ neural network inference engine built from scratch as a systems performance exercise. No frameworks, no dependencies — just raw matrix operations and compiler optimization.

## Status

**Month 1 complete.** Forward pass with binary weight loading, sub-millisecond latency, cache-aware benchmarking.

## Performance Summary

Benchmarks on i5-13H (compiled with `-O3 -march=native`):

| Metric | Latency |
|---|---|
| 128×128 matmul (p50) | 274 µs |
| 3-layer forward pass 128→64→32→10 (p50) | ~1 µs |
| Sequential memory access (1M floats) | 733 µs |
| Strided memory access (every 16th) | 5502 µs |
| Cache penalty (strided / sequential) | 7.5× |
| Output error vs golden reference | 3.58e-07 |

## Quick Start

```bash
# Build everything
make all

# Run the inference engine with external weights
make weights        # generate weights.bin (or use the included one)
./inference weights.bin

# Run benchmarks
./bench_bin         # 128×128 matmul p50/p95/p99
./cache_exp         # sequential vs strided access
./alignment_demo    # alignas(32) verification
```

Expected output from `./inference weights.bin`:
```
Forward pass: 0 µs  (p50=0  p95=0  p99=1)
Output: [0.2621, 0.2621, 0.2621, ...]
Max error vs golden: 3.5763e-07
```

## Build

```bash
make all          # Build all targets (optimized)
make debug        # Build with -O0 for debugging
make weights      # Regenerate weights.bin (requires Python 3 + numpy)
make clean        # Remove binaries
```

### Targets

| Command | Source | Description |
|---|---|---|
| `./engine` | `src/main.cpp` | Week 1: Tensor struct + matmul verification |
| `./enginev2` | `src/mainv2.cpp` | Week 2: 3-layer forward pass |
| `./inference` | `src/inference.cpp` | Week 4: Load weights.bin, run inference |
| `./bench_bin` | `benchmarks/bench.cpp` | 128×128 matmul latency distribution |
| `./cache_exp` | `benchmarks/cache_experiment.cpp` | Week 3: Cache line experiment |
| `./alignment_demo` | `benchmarks/alignment_demo.cpp` | Week 3: Memory alignment check |

## Project Structure

```
├── src/
│   ├── tensor.hpp / tensor.cpp    Tensor struct + matmul()
│   ├── Layer.hpp                  Fully connected layer (weights + bias)
│   ├── relu.hpp                   ReLU activation (max(0, x))
│   ├── main.cpp                   Week 1: matmul baseline test
│   ├── mainv2.cpp                 Week 2: 3-layer forward pass
│   └── inference.cpp              Week 4: binary weight loader + inference
├── benchmarks/
│   ├── bench.cpp                  128×128 matmul latency (p50/p95/p99)
│   ├── cache_experiment.cpp       Sequential vs strided memory access
│   └── alignment_demo.cpp         alignas(32) verification
├── scripts/
│   ├── generate_weights.py        Python weight generator (numpy)
│   └── golden.py                  Golden reference for validation
├── weights.bin                    Pretrained weights (binary)
├── golden_output.txt              Reference output for validation
└── Makefile                       Build system
```

## Month 1 Progress

| Week | Focus | Deliverable |
|---|---|---|
| **Week 1** | C++ foundations | `Tensor` struct, `matmul()`, chrono benchmarking |
| **Week 2** | Neural net forward pass | `Layer` struct, 3-layer ReLU network, end-to-end timing |
| **Week 3** | Memory & cache model | Cache line experiment, alignment demo, GitHub setup |
| **Week 4** | Binary I/O | Python weight generator, `./inference weights.bin`, golden validation |

## Roadmap

- **Month 1** — C++ foundations, forward pass, binary weights ✅
- **Month 2** — int8 quantization, latency percentiles, weight compression
- **Month 3** — AVX2 SIMD intrinsics, loop unrolling, cache blocking
- **Month 4** — CUDA kernel, GPU offloading

## License

MIT
