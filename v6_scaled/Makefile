CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS ?= -O3 -march=native -ffast-math -funroll-loops -std=c++17
NVFLAGS  ?= -O3 -std=c++17

all: v1_float32_exe v2_int8_exe v3_simd_exe v4_cuda_exe v4_cuda_shared_exe v4_cuda_batched_exe v4_cuda_pipeline_exe v5_transformer_exe main_cpu main_gpu

v1_float32_exe: v1_float32/main.cpp v1_float32/tensor.cpp
	$(CXX) $(CXXFLAGS) -o v1_float32_exe v1_float32/main.cpp v1_float32/tensor.cpp

v2_int8_exe: v2_int8/main.cpp
	$(CXX) $(CXXFLAGS) -o v2_int8_exe v2_int8/main.cpp

v3_simd_exe: v3_simd/main.cpp
	$(CXX) $(CXXFLAGS) -o v3_simd_exe v3_simd/main.cpp

v4_cuda_exe: v4_cuda/cuda_inference.cu
	$(NVCC) $(NVFLAGS) -o v4_cuda_exe v4_cuda/cuda_inference.cu

v4_cuda_shared_exe: v4_cuda/cuda_shared.cu
	$(NVCC) $(NVFLAGS) -o v4_cuda_shared_exe v4_cuda/cuda_shared.cu

v4_cuda_batched_exe: v4_cuda/cuda_batched.cu
	$(NVCC) $(NVFLAGS) -o v4_cuda_batched_exe v4_cuda/cuda_batched.cu

v4_cuda_pipeline_exe: v4_cuda/cuda_pipeline.cu
	$(NVCC) $(NVFLAGS) -o v4_cuda_pipeline_exe v4_cuda/cuda_pipeline.cu

v5_transformer_exe: v5_transformer/main_transformer.cpp v5_transformer/tensor.cpp
	$(CXX) $(CXXFLAGS) -o v5_transformer_exe v5_transformer/main_transformer.cpp v5_transformer/tensor.cpp

main_cpu: main_cpu.cpp
	$(CXX) $(CXXFLAGS) -o main_cpu main_cpu.cpp

main_gpu: main_gpu.cu
	$(NVCC) $(NVFLAGS) -o main_gpu main_gpu.cu

cpu: v1_float32_exe v2_int8_exe v3_simd_exe v5_transformer_exe main_cpu
	./main_cpu

gpu: all
	./main_gpu

v6_scaled_512: all
	./main_cpu 512
	./main_gpu 512

v6_scaled_1024: all
	./main_cpu 1024
	./main_gpu 1024

clean:
	rm -f v1_float32_exe v2_int8_exe v3_simd_exe v4_cuda_exe v4_cuda_shared_exe v4_cuda_batched_exe v4_cuda_pipeline_exe v5_transformer_exe main_cpu main_gpu cpu.txt gpu.txt

.PHONY: all clean cpu gpu v6_scaled_512 v6_scaled_1024
