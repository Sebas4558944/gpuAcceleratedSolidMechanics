
#include "componentKernels.h"

__global__ void axial(const real *F, const real _lambda, const real _mu, real *P, const int N)
{
    // Set the index for each thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    real Du11, Du22, Du33;
    real trace;

    if (index < N)
    {
        // Retrieve the diagonal components and subtract I
        Du11 = F[index] - 1.e0;
        Du22 = F[index + 4 * N] - 1.e0;
        Du33 = F[index + 8 * N] - 1.e0;

        // Compute the trace
        trace = (Du11 + Du22 + Du33) * _lambda;

        // Compute the diagonal components
        P[index] = trace + 2 * _mu * Du11;
        P[index + 4 * N] = trace + 2 * _mu * Du22;
        P[index + 8 * N] = trace + 2 * _mu * Du33;
    }

    return;
}

__global__ void axial(const real *F, const real _lambda, const real _mu, real *P, const int N, const int nLoop)
{
    // Set the index for each thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    real Du11, Du22, Du33;
    real trace;

    if (index < nLoop)
    {
        // Retrieve the diagonal components and subtract I
        Du11 = F[index] - 1.e0;
        Du22 = F[index + 4 * N] - 1.e0;
        Du33 = F[index + 8 * N] - 1.e0;

        // Compute the trace
        trace = (Du11 + Du22 + Du33) * _lambda;

        // Compute the diagonal components
        P[index] = trace + 2 * _mu * Du11;
        P[index + 4 * N] = trace + 2 * _mu * Du22;
        P[index + 8 * N] = trace + 2 * _mu * Du33;
    }

    return;
}

__global__ void shear(const real *F, const real _lambda, const real _mu, real *P, const int N, const int i, const int j)
{
    // Set the index for each thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    real Duij, Duji;

    // Retrieve the first off diagonal components and compute the stress
    if (index < N)
    {
        Duij = F[index + i * N];
        Duji = F[index + j * N];
        P[index + i * N] = P[index + j * N] = _mu * (Duij + Duji);
    }

    return;
}

__global__ void shear(const real *F, const real _lambda, const real _mu, real *P, const int N, const int i, const int j, const int nLoop)
{
    // Set the index for each thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    real Duij, Duji;

    // Retrieve the first off diagonal components and compute the stress
    if (index < nLoop)
    {
        Duij = F[index + i * N];
        Duji = F[index + j * N];
        P[index + i * N] = P[index + j * N] = _mu * (Duij + Duji);
    }

    return;
}