#ifndef COMPONENT_KERNELS_H
#define COMPONENT_KERNELS_H

#include <cuda_runtime.h>

using real = double;

__global__ void axial(const real *F, const real _lambda, const real _mu, real *P, const int N);

__global__ void axial(const real *F, const real _lambda, const real _mu, real *P, const int N, const int nLoop);

__global__ void shear(const real *F, const real _lambda, const real _mu, real *P, const int N, const int i, const int j);

__global__ void shear(const real *F, const real _lambda, const real _mu, real *P, const int N, const int i, const int j, const int nLoop);

#endif