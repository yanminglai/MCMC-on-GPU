#include <functional>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>

#define random_max ((double) RAND_MAX)
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

void metropolis_hastings(std::function<double (double[], int)> p, int num_samples, int dimension, double** samples);