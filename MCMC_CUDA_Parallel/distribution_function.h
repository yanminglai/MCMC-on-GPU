#include <math.h>

// 2D Gaussian, mean (ux,uy), sigma (sx,sy)
#define ux (0.0)
#define uy (0.0)
#define sx (1.0)
#define sy (1.0)

__device__
float distribution_function(float x[], int dim);