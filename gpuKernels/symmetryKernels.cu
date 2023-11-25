
#include "symmetryKernels.h"

__global__ void shear(const real *F, const real _lambda, const real _mu, real *Pij, const int N, const int i, const int j, int nLoop)
{
    // Set the index for each thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    real Duij, Duji;

    // Retrieve the first off diagonal components and compute the stress
    if (index < nLoop)
    {
        Duij = F[index + i * N];
        Duji = F[index + j * N];
        Pij[index + i * N] = _mu * (Duij + Duji);
    }

    return;
}