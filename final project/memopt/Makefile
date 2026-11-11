# Euler (RTX 4000 Ada): make
# other GPUs:           make ARCH=sm_120
ARCH ?= sm_89
NVCC ?= nvcc
FLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3
SRCS = main.cu $(wildcard v*.cu)

memopt: $(SRCS) memopt.cuh common.cuh
	$(NVCC) $(FLAGS) $(SRCS) -o memopt

clean:
	rm -f memopt

.PHONY: clean
