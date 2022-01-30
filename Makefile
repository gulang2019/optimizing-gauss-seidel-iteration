CC = gcc
NVCC = nvcc
NVCC_FLAGS = --gpu-architecture=sm_50 -std=c++11 -Xptxas -O3 -Wno-deprecated-gpu-targets -allow-unsupported-compiler
CFLAGS = -std=c99 -Ofast -march=native -fopenmp -Wall

LIBRARIES = -L/${CUDA_DIR}/lib64 -lcudart -lm

all: poisson_gpu

poisson_gpu:  poisson_gpu.o poisson_cpu.o
	$(CC) $(CFLAGS) $^ -o $@ $(LIBRARIES) 
poisson_cpu.o: poisson_gpu.c
	$(CC) $(CFLAGS) -c $^ -o $@
poisson_gpu.o: gauss_seidel.cu
	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@



clean:
	rm -f *.o poisson_gpu
