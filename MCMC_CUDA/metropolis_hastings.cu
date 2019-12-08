#include "metropolis_hastings.h"

/*
    dimension: length of parameter vector to the distribution function
*/
__device__
void _generate_initial_state(float initial_state[], int dimension, curandState_t prng_state) {
    for(int i=0; i<dimension; i++) {
        int random_number = curand(&prng_state);
        initial_state[i] = 2*float(random_number)/random_max - 1.0; //generate a number in [-1, 1]
    }
}

//Box muller method for sampling from Gaussian
__device__
double _rand_normal(float mean, float stddev, curandState_t prng_state)
{
    static float n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached)
    {
        float x, y, r;
        do
        {
            x = 2.0*curand(&prng_state)/RAND_MAX - 1;
            y = 2.0*curand(&prng_state)/RAND_MAX - 1;

            r = x*x + y*y;
        }
        while (r == 0.0 || r > 1.0);
        {
            float d = sqrt(-2.0*log(r)/r);
            float n1 = x*d;
            n2 = y*d;
            float result = n1*stddev + mean;
            n2_cached = 1;
            return result;
        }
    }
    else
    {
        n2_cached = 0;
        return n2*stddev + mean;
    }
}

__device__
void _propose_next_state(float current_state[], float next_state[], int dimension, curandState_t prng_state) {
    for(int i=0; i<dimension; i++) {
        next_state[i] = current_state[i] + _rand_normal(0.0, 1.0, prng_state); //random walk based on unit Gaussian output
    }
}

__device__
float _compute_acceptance_ratio(float current_state[], float proposed_state[]) {
    return distribution_function(proposed_state)/distribution_function(current_state);
}

/*
    p: the distribution function
    num_samples: total number of samples to be generated
    dimension: length of parameter vector to the distribution function
*/
__global__
void metropolis_hastings(int num_samples, int dimension, float** samples) {
    float* current_state = (float*)malloc(dimension * sizeof(float));
    //cudaMallocManaged(&current_state, dimension * sizeof(float));
    
    curandState_t prng_state;
    curand_init(0,0,0,&prng_state);
    _generate_initial_state(current_state, dimension, prng_state);

    for(int i=0; i<num_samples; i++) {
        _propose_next_state(current_state, samples[i], dimension, prng_state);
        float acceptance_ratio = MIN(1.0, _compute_acceptance_ratio(current_state, samples[i]));
        float test_probability = curand(&prng_state) / random_max;
        if(acceptance_ratio < test_probability) { //Reject, copy state forward
            for(int j=0; j<dimension; j++) {
                samples[i][j] = current_state[j];
            }
        }
        current_state = samples[i];
    }
}