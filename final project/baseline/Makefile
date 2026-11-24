# Euler: make   (g++ with OpenMP)
CXX ?= g++
FLAGS = -O3 -std=c++17 -fopenmp -Wall

bench: bench.cpp attention_cpu.cpp attention_cpu.h
	$(CXX) $(FLAGS) bench.cpp attention_cpu.cpp -o bench

clean:
	rm -f bench

.PHONY: clean
