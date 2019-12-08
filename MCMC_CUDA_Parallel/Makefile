CC=/usr/local/cuda/bin/nvcc

main: main.cu distribution_function.h distribution_function.cu metropolis_hastings.h metropolis_hastings.cu
	$(CC) -rdc=true main.cu distribution_function.cu metropolis_hastings.cu -o main.o
