#include "distribution_function.h"

// double distribution_function(double x[]) {
//     double exponent = -((x[0]-ux) * (x[0]-ux))/(2*sx*sx) - ((x[1]-uy) * (x[1]-uy))/(2*sy*sy);
//     double probability = exp(exponent);
//     return probability;
// }

double distribution_function(double x[], int dim) {
    double exponent = 0.0;
    for(int i=0; i<dim; i++) {
        exponent += -(x[i]*x[i])/2;
    }
    double probability = exp(exponent);
    return probability;
}