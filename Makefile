CXX = g++
OPT = -O2 -std=c++17

all: engine enginev2 cache_exp alignment_demo

engine:
	$(CXX) $(OPT) -o engine tensor.cpp main.cpp

enginev2:
	$(CXX) $(OPT) -o enginev2 tensor.cpp mainv2.cpp

cache_exp:
	$(CXX) $(OPT) -o cache_exp cache_experiment.cpp

alignment_demo:
	$(CXX) $(OPT) -o alignment_demo alignment_demo.cpp

clean:
	rm -f engine enginev2 cache_exp alignment_demo
