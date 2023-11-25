
#include <iostream>
#include "errorCheck.h"

void checkError(cudaError_t errSync, const char *func, const char *file, const int line)
{
    // Check for last Asynchronous error in CUDA
    cudaError_t errAsync = cudaGetLastError();
    if (errAsync != cudaSuccess)
    {
        std::cout << "CUDA Runtime Async Error at: " << file << ":" << line
                  << std::endl
                  << cudaGetErrorString(errAsync) << std::endl;

        std::exit(EXIT_FAILURE);
    }

    // Check for last Synchronous error in CUDA
    if (errSync != cudaSuccess)
    {
        std::cout << "CUDA Runtime Sync Error at: " << file << ":" << line << std::endl
                  << cudaGetErrorString(errSync) << " " << func << std::endl;

        std::exit(EXIT_FAILURE);
    }

    return;
}