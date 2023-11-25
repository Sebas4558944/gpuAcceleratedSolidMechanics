
#include "axialKernels.h"

__global__ void axial11(real *F, const real _lambda, const real _mu, real *P11, const int N, const int nLoop)
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
        trace = (Du22 + Du33) * _lambda;

        // Compute the diagonal components
        P11[index] = (2 * _mu + _lambda) * Du11 + trace;
    }

    return;
}

__global__ void axial22(real *F, const real _lambda, const real _mu, real *P22, const int N, const int nLoop)
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
        trace = (Du11 + Du33) * _lambda;

        // Compute the diagonal components
        P22[index] = (2 * _mu + _lambda) * Du22 + trace;
    }

    return;
}

__global__ void axial33(real *F, const real _lambda, const real _mu, real *P33, const int N, const int nLoop)
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
        trace = (Du11 + Du22) * _lambda;

        // Compute the diagonal components
        P33[index] = (2 * _mu + _lambda) * Du33 + trace;
    }

    return;
}