#ifndef GRADIENTS_H
#define GRADIENTS_H

using real = double;

void getCoalescedGradients(real F_pert, int N, real *F, int print);

void getStridedGradients(real F_pert, int N, real *F, int print);

void gradientsOrder2D(real F_pert, int N, real *F, int print);

void computeCG(real *F, int N, real *C, int print);

void computeCG(real *F, int N, real *C);

void computeCGLin(real *F, int N, real *C, int print);

void computeCGLin(real *F, int N, real *C);

#endif