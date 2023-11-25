
#include <cmath>
#include <iostream>
#include <cassert>

#include "verifyResults.h"

void verifyCoalescedResults(int N, real lambda, real mu, real *F, real *P, int print)
{
    // Create the reference stress to check values
    real *Pref, refTrace;
    Pref = (real *)malloc(N * 9 * sizeof(real));

    for (int i = 0; i < N; i++)
    {
        refTrace = (F[i] + F[i + N * 4] + F[i + N * 8] - 3.e0) * lambda;
        Pref[i] = refTrace + 2 * mu * (F[i] - 1.e0);
        Pref[i + N * 4] = refTrace + 2 * mu * (F[i + N * 4] - 1.e0);
        Pref[i + N * 8] = refTrace + 2 * mu * (F[i + N * 8] - 1.e0);

        Pref[i + N] = Pref[i + N * 3] = mu * (F[i + N] + F[i + N * 3]);
        Pref[i + N * 2] = Pref[i + N * 6] = mu * (F[i + N * 2] + F[i + N * 6]);
        Pref[i + N * 5] = Pref[i + N * 7] = mu * (F[i + N * 5] + F[i + N * 7]);
    }

    for (int i = 0; i < N; i++)
    {
        if (print == 1)
        {
            std::cout << "Computed PK-stress " << P[i] << " at index " << 0 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i] << " at index " << i << std::endl;
            std::cout << "Computed PK-stress " << P[i + N * 1] << " at index " << 1 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i + N * 1] << " at index " << i << std::endl;
            std::cout << "Computed PK-stress " << P[i + N * 2] << " at index " << 2 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i + N * 2] << " at index " << i << std::endl;
            std::cout << "Computed PK-stress " << P[i + N * 4] << " at index " << 4 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i + N * 4] << " at index " << i << std::endl;
            std::cout << "Computed PK-stress " << P[i + N * 5] << " at index " << 5 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i + N * 5] << " at index " << i << std::endl;
            std::cout << "Computed PK-stress " << P[i + N * 8] << " at index " << 8 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i + N * 8] << " at index " << i << std::endl;
        }

        assert(std::fabs(P[i] - Pref[i]) < 1.e0);
        assert(std::fabs(P[i + N * 1] - Pref[i + N * 1]) < 1.e0);
        assert(std::fabs(P[i + N * 2] - Pref[i + N * 2]) < 1.e0);
        assert(std::fabs(P[i + N * 4] - Pref[i + N * 4]) < 1.e0);
        assert(std::fabs(P[i + N * 5] - Pref[i + N * 5]) < 1.e0);
        assert(std::fabs(P[i + N * 8] - Pref[i + N * 8]) < 1.e0);
    }

    free(Pref);

    return;
}

void verifyIndexComp(int N, real lambda, real mu, real *F, real *P1, real *P2, real *P3, real *P5, real *P6, real *P9, int print)
{
    // Create the reference stress to check values
    real *Pref, refTrace;
    Pref = (real *)malloc(N * 9 * sizeof(real));

    for (int i = 0; i < N; i++)
    {
        refTrace = (F[i] + F[i + N * 4] + F[i + N * 8] - 3.e0) * lambda;
        Pref[i] = refTrace + 2 * mu * (F[i] - 1.e0);
        Pref[i + N * 4] = refTrace + 2 * mu * (F[i + N * 4] - 1.e0);
        Pref[i + N * 8] = refTrace + 2 * mu * (F[i + N * 8] - 1.e0);

        Pref[i + N] = Pref[i + N * 3] = mu * (F[i + N] + F[i + N * 3]);
        Pref[i + N * 2] = Pref[i + N * 6] = mu * (F[i + N * 2] + F[i + N * 6]);
        Pref[i + N * 5] = Pref[i + N * 7] = mu * (F[i + N * 5] + F[i + N * 7]);
    }

    for (int i = 0; i < N; i++)
    {
        if (print == 1)
        {
            std::cout << "Computed PK-stress " << P1[i] << " at index " << 0 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i] << " at index " << i << std::endl;
            std::cout << "Computed PK-stress " << P2[i] << " at index " << 1 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i + N * 1] << " at index " << i << std::endl;
            std::cout << "Computed PK-stress " << P3[i] << " at index " << 2 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i + N * 2] << " at index " << i << std::endl;
            std::cout << "Computed PK-stress " << P5[i] << " at index " << 4 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i + N * 4] << " at index " << i << std::endl;
            std::cout << "Computed PK-stress " << P6[i] << " at index " << 5 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i + N * 5] << " at index " << i << std::endl;
            std::cout << "Computed PK-stress " << P9[i] << " at index " << 8 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i + N * 8] << " at index " << i << std::endl;
        }

        assert(std::fabs(P1[i] - Pref[i]) < 1.e0);
        assert(std::fabs(P2[i] - Pref[i + N * 1]) < 1.e0);
        assert(std::fabs(P3[i] - Pref[i + N * 2]) < 1.e0);
        assert(std::fabs(P5[i] - Pref[i + N * 4]) < 1.e0);
        assert(std::fabs(P6[i] - Pref[i + N * 5]) < 1.e0);
        assert(std::fabs(P9[i] - Pref[i + N * 8]) < 1.e0);
    }

    free(Pref);

    return;
}

void verifyStridedResults(int N, real lambda, real mu, real *F, real *P, int print)
{
    // Create the reference stress to check values
    real *Pref, refTrace;
    Pref = (real *)malloc(N * 9 * sizeof(real));

    for (int i = 0; i < N; i++)
    {
        refTrace = (F[i * 9] + F[i * 9 + 4] + F[i * 9 + 8] - 3.e0) * lambda;
        Pref[i * 9] = refTrace + 2 * mu * (F[i * 9] - 1.e0);
        Pref[i * 9 + 4] = refTrace + 2 * mu * (F[i * 9 + 4] - 1.e0);
        Pref[i * 9 + 8] = refTrace + 2 * mu * (F[i * 9 + 8] - 1.e0);

        Pref[i * 9 + 1] = Pref[i * 9 + 3] = mu * (F[i * 9 + 1] + F[i * 9 + 3]);
        Pref[i * 9 + 2] = Pref[i * 9 + 6] = mu * (F[i * 9 + 2] + F[i * 9 + 6]);
        Pref[i * 9 + 5] = Pref[i * 9 + 7] = mu * (F[i * 9 + 5] + F[i * 9 + 7]);
    }

    for (int i = 0; i < N; i++)
    {
        if (print == 1)
        {
            std::cout << "Computed PK-stress " << P[i * 9] << " at index " << 0 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i * 9] << " at index " << i << std::endl;
            std::cout << "Computed PK-stress " << P[i * 9 + 1] << " at index " << 1 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i * 9 + 1] << " at index " << i << std::endl;
            std::cout << "Computed PK-stress " << P[i * 9 + 2] << " at index " << 2 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i * 9 + 2] << " at index " << i << std::endl;
            std::cout << "Computed PK-stress " << P[i * 9 + 4] << " at index " << 4 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i * 9 + 4] << " at index " << i << std::endl;
            std::cout << "Computed PK-stress " << P[i * 9 + 5] << " at index " << 5 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i * 9 + 5] << " at index " << i << std::endl;
            std::cout << "Computed PK-stress " << P[i * 9 + 8] << " at index " << 8 << std::endl;
            std::cout << "Reference PK-stress " << Pref[i * 9 + 8] << " at index " << i << std::endl;
        }

        assert(std::fabs(P[i * 9] - Pref[i * 9]) < 1.e0);
        assert(std::fabs(P[i * 9 + 1] - Pref[i * 9 + 1]) < 1.e0);
        assert(std::fabs(P[i * 9 + 2] - Pref[i * 9 + 2]) < 1.e0);
        assert(std::fabs(P[i * 9 + 4] - Pref[i * 9 + 4]) < 1.e0);
        assert(std::fabs(P[i * 9 + 5] - Pref[i * 9 + 5]) < 1.e0);
        assert(std::fabs(P[i * 9 + 8] - Pref[i * 9 + 8]) < 1.e0);
    }

    free(Pref);

    return;
}
