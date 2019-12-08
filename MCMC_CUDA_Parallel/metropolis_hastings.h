#include <functional>
#include <math.h>
#include <iostream>

#include <curand.h>
#include <curand_kernel.h>

#include "distribution_function.h"

#define random_max ((float) RAND_MAX)
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

__global__
void metropolis_hastings(int num_samples, int dimension, float** samples);