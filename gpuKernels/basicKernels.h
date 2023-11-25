#ifndef BASICKERNELS_H
#define BASICKERNELS_H

#include <cuda_runtime.h>

using real = double;

__global__ void stridedConstitutive(const real *F, const real _lambda, const real _mu, real *P, const int N);

__global__ void coalescedConstitutive(const real *F, const real _lambda, const real _mu, real *P, const int N);

__global__ void splitConstitutive(const real *F, const real _lambda, const real _mu, real *P, const int N);

__global__ void componentConstitutive(const real *F, const real _lambda, const real _mu, real *P, const int N);

#endif
