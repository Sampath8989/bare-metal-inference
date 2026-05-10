# low-latency-inference

Building a neural net inference engine from scratch in C++. No frameworks.

## Baseline numbers (i5-13H, -O2)


| | latency |
|---|---|
| 128×128 matmul | ~1200 us |
| 3-layer forward pass (128→64→32→10) | ~19 us |
| cache sequential vs strided | 5.8x difference |
| aligned % 32 | always 0 |
| unaligned % 32 | random (not guaranteed) |

## Build

```bash
make all
./engine          # matmul baseline
./enginev2        # forward pass
./cache_exp       # cache experiment
./alignment_demo  # alignment check
```

## Structure

```
tensor.hpp / tensor.cpp    Tensor struct + matmul
Layer.hpp                  fully connected layer
relu.hpp                   ReLU activation
main.cpp                   Week 1: matmul baseline
mainv2.cpp                 Week 2: 3-layer forward pass
cache_experiment.cpp       Week 3: sequential vs strided
alignment_demo.cpp         Week 3: alignas(32) demo
```

## Roadmap

- Month 1 — C++ foundations, forward pass ✅
- Month 2 — int8 quantisation, latency percentiles
- Month 3 — AVX2 SIMD
- Month 4 — CUDA
