CXX     = g++
CXXFLAGS = -O3 -march=native -ffast-math -funroll-loops -std=c++17
DEBUGFLAGS = -O0 -std=c++17
INC     = -Isrc

SRC     = src/tensor.cpp src/main.cpp
SRC_V2  = src/tensor.cpp src/mainv2.cpp
INF_SRC = src/tensor.cpp src/inference.cpp
BENCH_SRC = src/tensor.cpp benchmarks/bench.cpp
CACHE_SRC = benchmarks/cache_experiment.cpp
ALIGN_SRC = benchmarks/alignment_demo.cpp

# float32 baseline build
V1_SRC = v1_float32/main.cpp src/tensor.cpp
# int8 quantized build
V2_SRC = v2_int8/main.cpp
BENCH_BOTH_SRC = benchmarks/bench_both.cpp src/tensor.cpp

# AVX2 SIMD versioned builds
V3_BENCH_SRC = v3_simd/bench_simd.cpp src/tensor.cpp
V3_MAIN_SRC = v3_simd/main.cpp
V3_VAL_SRC = v3_simd/validate.cpp

all: engine engine_debug enginev2 enginev2_debug bench_bin inference cache_exp alignment_demo float32_v1 int8 bench_int8 v3_bench v3_simd_main v3_validate

engine:
	$(CXX) $(CXXFLAGS) $(INC) -o engine $(SRC)

engine_debug:
	$(CXX) $(DEBUGFLAGS) $(INC) -o engine_debug $(SRC)

enginev2:
	$(CXX) $(CXXFLAGS) $(INC) -o enginev2 $(SRC_V2)

enginev2_debug:
	$(CXX) $(DEBUGFLAGS) $(INC) -o enginev2_debug $(SRC_V2)

bench_bin:
	$(CXX) $(CXXFLAGS) $(INC) -o bench_bin $(BENCH_SRC)

inference:
	$(CXX) $(CXXFLAGS) $(INC) -o inference $(INF_SRC)

cache_exp:
	$(CXX) $(CXXFLAGS) -o cache_exp $(CACHE_SRC)

alignment_demo:
	$(CXX) $(CXXFLAGS) -o alignment_demo $(ALIGN_SRC)

float32_v1:
	$(CXX) $(CXXFLAGS) $(INC) -o float32_v1 $(V1_SRC)

int8:
	$(CXX) $(CXXFLAGS) $(INC) -o int8 $(V2_SRC)

bench_int8:
	$(CXX) $(CXXFLAGS) $(INC) -o bench_int8 $(BENCH_BOTH_SRC)

v3_bench:
	$(CXX) $(CXXFLAGS) $(INC) -o v3_bench $(V3_BENCH_SRC)

v3_simd_main:
	$(CXX) $(CXXFLAGS) $(INC) -o v3_simd_main $(V3_MAIN_SRC)

v3_validate:
	$(CXX) $(CXXFLAGS) $(INC) -o v3_validate $(V3_VAL_SRC)

weights:
	python3 scripts/generate_weights.py

clean:
	rm -f engine engine_debug enginev2 enginev2_debug bench_bin inference cache_exp alignment_demo float32_v1 int8 bench_int8 v3_bench v3_simd_main v3_validate

.PHONY: all clean weights inference cache_exp alignment_demo bench_bin float32_v1 int8 bench_int8 v3_bench v3_simd_main v3_validate
