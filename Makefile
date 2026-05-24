CXX = g++
OPT  = -O2 -std=c++17 -Isrc
DEBUG = -O0 -std=c++17 -Isrc

SRC     = src/tensor.cpp src/main.cpp
SRC_V2  = src/tensor.cpp src/mainv2.cpp
INF_SRC = src/tensor.cpp src/inference.cpp
BENCH_SRC = src/tensor.cpp benchmarks/bench.cpp
CACHE_SRC = benchmarks/cache_experiment.cpp
ALIGN_SRC = benchmarks/alignment_demo.cpp

all: engine enginev2 inference bench_bin cache_exp alignment_demo

engine:
	$(CXX) $(OPT) -o engine $(SRC)

engine_debug:
	$(CXX) $(DEBUG) -o engine_debug $(SRC)

enginev2:
	$(CXX) $(OPT) -o enginev2 $(SRC_V2)

inference:
	$(CXX) $(OPT) -o inference $(INF_SRC)

bench_bin:
	$(CXX) $(OPT) -o bench_bin $(BENCH_SRC)

cache_exp:
	$(CXX) $(OPT) -o cache_exp $(CACHE_SRC)

alignment_demo:
	$(CXX) $(OPT) -o alignment_demo $(ALIGN_SRC)

weights:
	python3 scripts/generate_weights.py

clean:
	rm -f engine engine_debug enginev2 inference bench_bin cache_exp alignment_demo

.PHONY: all clean weights
