
#include "basicKernels.h"

__global__ void stridedConstitutive(const real *F, const real _lambda, const real _mu, real *P, const int N)
{
    // Set the index for each thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute gradient of deformation
    real Du[9];

    int i, j;

    if (index < N)
    {
        for (i = 0; i < 9; i++)
        {
            Du[i] = F[index * 9 + i];
        }
        for (j = 0; j < 3; j++)
        {
            Du[j * 3 + j] -= 1.e0;
        }

        // Compute Strains
        real trace = Du[0] + Du[4] + Du[8];

        // Compute the PK stress for the strain point
        P[index * 9] = _lambda * trace + 2. * _mu * Du[0];
        P[index * 9 + 4] = _lambda * trace + 2. * _mu * Du[4];
        P[index * 9 + 8] = _lambda * trace + 2. * _mu * Du[8];

        P[index * 9 + 1] = P[index * 9 + 3] = _mu * (Du[1] + Du[3]);
        P[index * 9 + 2] = P[index * 9 + 6] = _mu * (Du[2] + Du[6]);
        P[index * 9 + 5] = P[index * 9 + 7] = _mu * (Du[5] + Du[7]);
    }

    return;
}

__global__ void coalescedConstitutive(const real *F, const real _lambda, const real _mu, real *P, const int N)
{
    // Set the index for each thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute gradient of deformation
    real Du[9];

    int i, j;

    if (index < N)
    {
        for (i = 0; i < 9; i++)
        {
            Du[i] = F[index + i * N];
        }
        for (j = 0; j < 3; j++)
        {
            Du[j * 3 + j] -= 1.e0;
        }

        // Compute Strains
        real trace = Du[0] + Du[4] + Du[8];

        // Compute the PK stress for the strain point
        P[index] = _lambda * trace + 2. * _mu * Du[0];
        P[index + 4 * N] = _lambda * trace + 2. * _mu * Du[4];
        P[index + 8 * N] = _lambda * trace + 2. * _mu * Du[8];

        P[index + 1 * N] = P[index + 3 * N] = _mu * (Du[1] + Du[3]);
        P[index + 2 * N] = P[index + 6 * N] = _mu * (Du[2] + Du[6]);
        P[index + 5 * N] = P[index + 7 * N] = _mu * (Du[5] + Du[7]);
    }

    return;
}

__global__ void splitConstitutive(const real *F, const real _lambda, const real _mu, real *P, const int N)
{
    // Set the index for each thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute gradient of deformation
    real Du[9];

    if (index < N)
    {
        Du[0] = F[index] - 1.;
        Du[4] = F[index + N * 4] - 1.;
        Du[8] = F[index + N * 8] - 1.;

        real trace = Du[0] + Du[4] + Du[8];

        P[index] = _lambda * trace + 2. * _mu * Du[0];
    }
    else if (index < (2 * N))
    {
        Du[1] = F[index];
        Du[3] = F[index + N * 2];

        P[index] = _mu * (Du[1] + Du[3]);
    }
    else if (index < (3 * N))
    {
        Du[2] = F[index];
        Du[6] = F[index + N * 4];

        P[index] = _mu * (Du[2] + Du[6]);
    }
    else if (index < (4 * N))
    {
        Du[1] = F[index - N * 2];
        Du[3] = F[index];

        P[index] = _mu * (Du[1] + Du[3]);
    }
    else if (index < (5 * N))
    {
        Du[0] = F[index - 4 * N] - 1.;
        Du[4] = F[index] - 1.;
        Du[8] = F[index + N * 4] - 1.;

        real trace = Du[0] + Du[4] + Du[8];

        P[index] = _lambda * trace + 2. * _mu * Du[4];
    }
    else if (index < (6 * N))
    {
        Du[5] = F[index];
        Du[7] = F[index + N * 2];

        P[index] = _mu * (Du[5] + Du[7]);
    }
    else if (index < (7 * N))
    {
        Du[2] = F[index - N * 4];
        Du[6] = F[index];

        P[index] = _mu * (Du[2] + Du[6]);
    }
    else if (index < (8 * N))
    {
        Du[5] = F[index - N * 2];
        Du[7] = F[index];

        P[index] = _mu * (Du[5] + Du[7]);
    }
    else if (index < (9 * N))
    {
        Du[0] = F[index - 8 * N] - 1.;
        Du[4] = F[index - N * 4] - 1.;
        Du[8] = F[index] - 1.;

        real trace = Du[0] + Du[4] + Du[8];

        P[index] = _lambda * trace + 2. * _mu * Du[8];
    }

    return;
}

__global__ void componentConstitutive(const real *F, const real _lambda, const real _mu, real *P, const int N)
{
    // Set the index for each thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
    {
        real Du1 = F[index] - 1.;
        real Du2 = F[index + 4 * N] - 1.;
        real Du3 = F[index + 8 * N] - 1.;

        // Compute the trace
        real trace = (Du1 + Du2 + Du3) * _lambda;

        // Compute the diagonal components
        P[index] = trace + 2 * _mu * (Du1);
        P[index + 4 * N] = trace + 2 * _mu * (Du2);
        P[index + 8 * N] = trace + 2 * _mu * (Du3);

        // Retrieve the first off diagonal components and compute the stress
        Du1 = F[index + N];
        Du2 = F[index + 3 * N];
        P[index + N] = P[index + 3 * N] = _mu * (Du1 + Du2);

        // Retrieve the second off diagonal components and compute the stress
        Du1 = F[index + 2 * N];
        Du2 = F[index + 6 * N];
        P[index + 2 * N] = P[index + 6 * N] = _mu * (Du1 + Du2);

        // Retrieve the third off diagonal components and compute the stress
        Du1 = F[index + 5 * N];
        Du2 = F[index + 7 * N];
        P[index + 5 * N] = P[index + 7 * N] = _mu * (Du1 + Du2);
    }

    return;
}
