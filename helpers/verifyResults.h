#ifndef VERIFY_RESULTS_H
#define VERIFY_RESULTS_H

using real = double;

void verifyCoalescedResults(int N, real lambda, real mu, real *C, real *P, int print);

void verifyIndexComp(int N, real lambda, real mu, real *F, real *P1, real *P2, real *P3, real *P5, real *P6, real *P9, int print);

void verifyStridedResults(int N, real lambda, real mu, real *C, real *P, int print);

#endif