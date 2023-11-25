#ifndef SYMMETRY_KERNELS_H
#define SYMMETRY_KERNELS_H

#include <cuda_runtime.h>

using real = double;

__global__ void shear(const real *F, const real _lambda, const real _mu, real *Pij, const int N, const int i, const int j, int loopSize);

#endif
