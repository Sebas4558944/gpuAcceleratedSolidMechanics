#ifndef ERRORCHECK_H
#define ERRORCHECK_H

#include <cuda_runtime.h>

void checkError(cudaError_t errSync, const char *func, const char *file, const int line);

#endif