#ifndef AXIAL_KERNELS_H
#define AXIAL_KERNELS_H

#include <cuda_runtime.h>

using real = double;

__global__ void axial11(real *F, const real _lambda, const real _mu, real *P11, const int N, const int nLoop);

__global__ void axial22(real *F, const real _lambda, const real _mu, real *P22, const int N, const int nLoop);

__global__ void axial33(real *F, const real _lambda, const real _mu, real *P33, const int N, const int nLoop);

#endif