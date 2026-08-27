# Bare-Metal GPU Optimization Benchmark

This directory contains a clean, standalone GPU matrix multiplication optimization benchmark designed for **NVIDIA T4** (and modern CUDA GPUs).

---

## Benchmark Configurations

The benchmark evaluates six configurations across matrix sizes `128×128`, `256×256`, `512×512`, `1024×1024`, and `2048×2048`:

1. **Baseline**: Naive global memory FP32 CUDA kernel.
2. **FP16**: Native half-precision arithmetic (`half`) on standard CUDA cores.
3. **Shared Memory**: $16\times 16$ 2D tiled on-chip caching to minimize global memory roundtrips.
4. **WMMA Tensor Cores**: $16\times 16\times 16$ hardware matrix-multiply-accumulate via `nvcuda::wmma`.
5. **Kernel Fusion**: Fused GEMM + ReLU activation epilogue (eliminating intermediate DRAM roundtrips).
6. **Combined**: **REAL integrated kernel** uniting **FP16 + Shared Memory Tiling ($64\times 64$, `SMEM_PAD=8`) + WMMA Tensor Cores + Fused Activation Epilogue**.

---

## File Structure

```text
v12_combined/
├── baseline.cu        # 1. Baseline naive FP32 kernel
├── fp16.cu            # 2. FP16 half-precision kernel
├── shared_memory.cu   # 3. 16x16 shared memory tiled FP32 kernel
├── wmma.cu            # 4. 16x16x16 WMMA Tensor Core kernel
├── kernel_fusion.cu   # 5. Fused GEMM + activation epilogue kernel
├── combined.cu        # 6. Real integrated combined GPU kernel
├── main.cu            # Master benchmark driver & result generator
├── plot_results.py    # Dynamic chart generation from gpu_results.csv
├── run_colab.sh       # Automated all-in-one execution script for Colab
└── README.md          # Documentation and execution instructions
```

---

## Google Colab Instructions (NVIDIA T4)

### 1. Set Hardware Accelerator
1. In the Colab menu, select **Runtime** → **Change runtime type**.
2. Select **T4 GPU** under Hardware accelerator and click **Save**.

### 2. Verify GPU Environment
```python
!nvidia-smi
!nvcc --version
```

### 3. Navigate to `v12_combined/`
```python
%cd v12_combined
```

### 4. Compile the Benchmark
```python
!nvcc -O3 -arch=sm_75 -std=c++17 main.cu -o benchmark_gpu
```
*(Note: `-arch=sm_75` targets the Turing architecture of NVIDIA T4. For Ampere cards such as RTX 2050 / RTX 30-series, use `-arch=sm_86`)*.

### 5. Run the Benchmark
```python
!./benchmark_gpu
```
This will print live timing to the terminal and generate:
- `gpu_results.csv`
- `gpu_results.txt`

### 6. Generate and Display Charts
```python
!python3 plot_results.py

from IPython.display import Image, display
display(Image("gpu_all_optimizations.png"))
display(Image("gpu_baseline_vs_combined.png"))
display(Image("gpu_speedup.png"))
```

---

## All-In-One Script
Alternatively, run the complete automated pipeline with:
```bash
bash run_colab.sh
```
