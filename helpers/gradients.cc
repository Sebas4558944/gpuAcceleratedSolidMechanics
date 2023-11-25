#include <iostream>

#include "gradients.h"

/**
 * Function to compute a certain amount of random deformation gradients
 *
 * @param[in] F_pert perturbation applied
 * @param[in] N the amount of deformation gradients generated
 * @param[in, out] F array containing all deformation gradients
 * @param[in] print determines whether intermediate results are printed
 */
void getCoalescedGradiets(real F_pert, int N, real *F, int print)
{
    // Loop over all components of the deformation gradients
    for (int i = 0; i < (N * 9); i++)
    {
        // Compute the random perturbations
        F[i] = F_pert * (2 * (real)rand() / RAND_MAX - 1.0);

        // Add identity to the random perturbation
        if (i < N)
        {
            F[i] += 1.0;
        }
        else if (i < (N * 5))
        {
            if (i >= (N * 4))
            {
                F[i] += 1.0;
            }
        }
        else
        {
            if (i >= (N * 8))
            {
                F[i] += 1.0;
            }
        }
    }

    // Print the resulting deformation tensor
    if (print == 1)
    {
        std::cout << "\nDeformation tensor send to the GPU: " << std::endl;
        for (int i = 0; i < (N * 9); i++)
        {
            std::cout << "Value at index " << i << " is equal to:   " << F[i] << std::endl;
        }
    }

    return;
}

void getStridedGradients(real F_pert, int N, real *F, int print)
{
    // Loop over all components of the deformation gradients
    int i1 = 0, i5 = 4, i9 = 8;

    for (int i = 0; i < (N * 9); i++)
    {
        // Compute the random perturbations
        F[i] = F_pert * (2 * (real)rand() / RAND_MAX - 1.0);

        // Add identity to the random perturbation
        if (i == i1)
        {
            F[i] += 1.0;
            i1 += 9;
        }
        else if (i == i5)
        {
            F[i] += 1.0;
            i5 += 9;
        }
        else if (i == i9)
        {
            F[i] += 1.0;
            i9 += 9;
        }
    }

    // Print the resulting deformation tensor
    if (print == 1)
    {
        std::cout << "\nDeformation tensor send to the GPU: " << std::endl;
        for (int i = 0; i < (N * 9); i++)
        {
            std::cout << "Value at index " << i << " is equal to:   " << F[i] << std::endl;
        }
    }

    return;
}

void gradientsOrder2D(real F_pert, int N, real *F, int print)
{
    // Loop over all components of the deformation gradients

    for (int i = 0; i < (4 * N); i++)
    {
        // Compute the random perturbations
        F[i] = F_pert * (2 * (real)rand() / RAND_MAX - 1.0);
    }

    for (int i = 0; i < N; i++)
    {
        // Compute the random perturbations
        F[i] += 1;
    }

    for (int i = 0; i < N; i++)
    {
        // Compute the random perturbations
        F[i + 3 * N] += 1;
    }

    // Print the resulting deformation tensor
    if (print == 1)
    {
        std::cout << "\nDeformation tensor send to the GPU: " << std::endl;
        for (int i = 0; i < (N * 4); i++)
        {
            std::cout << "Value at index " << i << " is equal to:   " << F[i] << std::endl;
        }
    }

    return;
}

/**
 * Compute the right Cauchy-Green tensor
 *
 * @param[in] F the array with deformation gradients
 * @param[in] N the amount of deformation gradients
 * @param[in, out] C the array with right Cauchy-Green strain tensors
 * @param[in] print determines whether intermediate results are printed
 */
void computeCG(real *F, int N, real *C, int print)
{
    // Set up transpose of the deformation gradient
    real FT[9];

    for (int i = 0; i < N; i++)
    {

        // Compute the transposed F, j goes column-wise
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
                FT[j + k * 3] = F[i + (j * 3 + k) * N];
        }

        // Compute the right-Cauchy Green
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                C[i + (k + j * 3) * N] = 0.0;
                for (int l = 0; l < 3; l++)
                {
                    C[i + (k + j * 3) * N] += FT[j * 3 + l] * F[i + (l * 3 + k) * N];

                    if (print == 1)
                    {
                        std::cout << "\nFT component: " << FT[j * 3 + l] << std::endl
                                  << "FT index: " << j * 3 + l << std::endl
                                  << "F component: " << F[i + (l * 3 + k) * N] << std::endl
                                  << "F index: " << i + (l * 3 + k) * N << std::endl
                                  << "C component: " << C[i + (k + j * 3) * N] << std::endl
                                  << "C index: " << i + (k + j * 3) * N << std::endl;
                    }
                }
            }
        }
    }

    // Print the resulting right-Cauchy Green tensor
    if (print == 1)
    {
        std::cout << "\nRight-Cauchy Green tensor computed: " << std::endl;
        for (int i = 0; i < (N * 9); i++)
        {
            std::cout << "Value at index " << i << " is equal to:   " << C[i] << std::endl;
        }
    }

    return;
}

void computeCG(real *F, int N, real *C)
{
    // Set up transpose of the deformation gradient
    real FT[9];

    for (int i = 0; i < N; i++)
    {

        // Compute the transposed F, j goes column-wise
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
                FT[j + k * 3] = F[i * 9 + j * 3 + k];
        }

        // Compute the right-Cauchy Green
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                C[i * 9 + k + j * 3] = 0.0;
                for (int l = 0; l < 3; l++)
                {
                    C[i * 9 + k + j * 3] += FT[j * 3 + l] * F[i * 9 + l * 3 + k];
                }
            }
        }
    }

    return;
}

/**
 * Compute the right Cauchy-Green tensor
 *
 * @param[in] F the array with deformation gradients
 * @param[in] N the amount of deformation gradients
 * @param[in, out] C the array with right Cauchy-Green strain tensors
 * @param[in] print determines whether intermediate results are printed
 */
void computeCGLin(real *F, int N, real *C, int print)
{
    // Set up transpose of the deformation gradient
    real FT[9];
    real _F[9];

    for (int i = 0; i < N; i++)
    {

        // Compute the transposed F, j goes column-wise
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                FT[j + k * 3] = F[i + (j * 3 + k) * N];
                _F[j * 3 + k] = F[i + (j * 3 + k) * N];

                // Subtract only from the transpose identity as you sum for elastic so C = Du + DuT + I and not Du + I + DuT + I
                if (j == k)
                {
                    FT[j + k * 3] -= 1.0;
                }
            }
        }

        // Compute the right-Cauchy Green
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {

                C[i + (k + j * 3) * N] = FT[k + j * 3] + _F[k + j * 3];

                if (print == 1)
                {
                    std::cout << "\nFT component: " << FT[k + j * 3] << std::endl
                              << "FT index: " << k + j * 3 << std::endl
                              << "F component: " << _F[k + j * 3] << std::endl
                              << "F index: " << k + j * 3 << std::endl
                              << "C component: " << C[i + (k + j * 3) * N] << std::endl
                              << "C index: " << i + (k + j * 3) * N << std::endl;
                }
            }
        }
    }

    // Print the resulting right-Cauchy Green tensor
    if (print == 1)
    {
        std::cout << "\nRight-Cauchy Green tensor computed: " << std::endl;
        for (int i = 0; i < (N * 9); i++)
        {
            std::cout << "Value at index " << i << " is equal to:   " << C[i] << std::endl;
        }
    }

    return;
}

void computeCGLin(real *F, int N, real *C)
{
    // Set up transpose of the deformation gradient
    real FT[9];
    real _F[9];

    for (int i = 0; i < N; i++)
    {

        // Compute the transposed F, j goes column-wise
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                FT[j + k * 3] = F[i * 9 + j * 3 + k];
                _F[j * 3 + k] = F[i * 9 + j * 3 + k];

                // Subtract only from the transpose identity as you sum for elastic so C = Du + DuT + I and not Du + I + DuT + I
                if (j == k)
                {
                    FT[j + k * 3] -= 1.0;
                }
            }
        }

        // Compute the right-Cauchy Green
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {

                C[i * 9 + k + j * 3] = FT[k + j * 3] + _F[k + j * 3];
            }
        }
    }

    return;
}