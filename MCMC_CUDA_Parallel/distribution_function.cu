#include "distribution_function.h"

// __device__
// float distribution_function(float x[]) {
//     float exponent = -((x[0]-ux) * (x[0]-ux))/(2*sx*sx) - ((x[1]-uy) * (x[1]-uy))/(2*sy*sy);
//     float probability = exp(exponent);
//     return probability;
// }

__device__
float distribution_function(float x[], int dim) {
    float exponent = 0.0;
    for(int i=0; i<dim; i++) {
        exponent += -(x[i]*x[i])/2;
    }
    float probability = exp(exponent);
    return probability;
}