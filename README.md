## Contribution
Final Project for CS 550 Computer Architecture
Team member: Webster Bei, Yanming Lai, Joey Liang

## Introduction
This repository contains code for generating random samples according to a given distribution function using the independent Metropolis-Hastings algorithm. There are three different sub-directories in this repository.  
* MCMC_CPU: implementation on CPU
* MCMC_CUDA: implementation on CUDA, using only a single thread from a single block
* MCMC_CUDA_Parallel: implementation on CUDA, one can specify the number of blocks and number of threads per block through the commandline input

## Usage
### MCMC_CPU
Compile
```
make
```
Run
```
./main.o [number_of_samples]
```
### MCMC_CUDA
Compile
```
make
```
Run
```
./main.o [number_of_samples]
```
### MCMC_CUDA_Parallel
Compile
```
make
```
Run
```
./main.o [number_of_samples] [number_of_blocks] [number_of_threads_per_block]
```

## Output
The programs will output a *samples.csv* file which contains all the random samples generated. There will also be console output that details the amount of time that the program spent on memory allocation and actual computation.

