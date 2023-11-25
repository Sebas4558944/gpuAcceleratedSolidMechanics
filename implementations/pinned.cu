// Include the header files needed
#include <iostream>

#include "../gpuKernels/basicKernels.h"
#include "../gpuKernels/componentKernels.h"
#include "../helpers/errorCheck.h"
#include "../helpers/verifyResults.h"
#include "../helpers/gradients.h"

int main(int argc, char **argv)
{
    // Initialize cuda error object
    cudaError_t status;

    // Reset the device before starting
    status = cudaDeviceReset();
    checkError(status, "cudaDeviceReset", __FILE__, __LINE__);

    // Print results of input data and verification
    int print = 0;

    // Add random material properties - copy to constant memory
    real lambda = 55.27e9;
    real mu = 25.95e9;

    std::cout << "\n---------- Constitutive update ----------" << std::endl
              << "Material properties: " << std::endl
              << "Lambda value:     " << lambda << std::endl
              << "Mu value:         " << mu << "\n"
              << std::endl;

    // Determine the number of elements and quadrature points per element
    int elem = atoi(argv[1]) * 1e6;
    int quads = 1;
    int nEval = elem * quads;

    // Log how many iterations are performed
    std::cout << "\n---------- Launch Configuration ----------" << std::endl
              << "Number of quadrature points:      " << quads << std::endl
              << "Number of elements :              " << elem / 1e6 << " M " << std::endl
              << "Number of constitutive updates:   " << nEval / 1e6 << " M" << std::endl;

    // Allocate workers
    int blockSize, gridSize, lastUse;
    if (nEval > 512)
    {
        blockSize = 512;
        gridSize = nEval / blockSize + 1;
        lastUse = nEval - (gridSize - 1) * blockSize;
        if (lastUse == 0)
        {
            gridSize -= 1;
            lastUse = blockSize;
        }
    }
    else
    {
        blockSize = nEval;
        gridSize = 1;
        lastUse = blockSize;
    }

    // Show a summary of the grid that is used and its allocated threads
    std::cout << "\nNumber of threads per block:      " << blockSize << std::endl
              << "Number of blocks:                 " << gridSize << std::endl
              << "Number of threads:                " << gridSize * blockSize << std::endl
              << "Active threads in last block:     " << lastUse << std::endl;

    // Initialize variables to be stored in the GPU
    real *F, *P, *Fg, *Pg;

    // Allocate CPU memory
    status = cudaMallocHost(&F, nEval * 9 * sizeof(real));
    checkError(status, "cudaMallocHost", __FILE__, __LINE__);
    status = cudaMallocHost(&P, nEval * 9 * sizeof(real));
    checkError(status, "cudaMallocHost", __FILE__, __LINE__);

    // Allocate GPU memory
    status = cudaMalloc(&Fg, nEval * 9 * sizeof(real));
    checkError(status, "cudaMalloc", __FILE__, __LINE__);
    status = cudaMalloc(&Pg, nEval * 9 * sizeof(real));
    checkError(status, "cudaMalloc", __FILE__, __LINE__);

    // Define an an array of random deformation gradients as a perturbation of the identity
    real F_pert = 0.1;
    getCoalescedGradients(F_pert, nEval, F, print);

    // Create cuda timers
    cudaEvent_t startComp, stopComp, startSend, stopSend;
    status = cudaEventCreate(&startComp);
    checkError(status, "cudaEventCreate", __FILE__, __LINE__);
    status = cudaEventCreate(&stopComp);
    checkError(status, "cudaEventCreate", __FILE__, __LINE__);
    status = cudaEventCreate(&startSend);
    checkError(status, "cudaEventCreate", __FILE__, __LINE__);
    status = cudaEventCreate(&stopSend);
    checkError(status, "cudaEventCreate", __FILE__, __LINE__);

    std::cout << "\n---------- Cuda Event Timing ----------" << std::endl;

    // Sending memory and timing the transaction
    status = cudaEventRecord(startSend);
    checkError(status, "cudaEventRecord", __FILE__, __LINE__);

    cudaMemcpy(Fg, F, nEval * 9 * sizeof(real), cudaMemcpyHostToDevice);

    status = cudaEventRecord(stopSend);
    checkError(status, "cudaEventRecord", __FILE__, __LINE__);
    status = cudaEventSynchronize(stopSend);
    checkError(status, "cudaEventSynchronize", __FILE__, __LINE__);

    // Check the timing
    float milliseconds = 0.;
    status = cudaEventElapsedTime(&milliseconds, startSend, stopSend);
    checkError(status, "cudaEventElapsedTime", __FILE__, __LINE__);

    std::cout << "\nSending memory took: " << milliseconds << " ms" << std::endl;

    // Executing computation
    status = cudaEventRecord(startComp);
    checkError(status, "cudaEventRecord", __FILE__, __LINE__);

    std::cout << "\nSynchronizing device..." << std::endl;

    // Running the coalesced pinned memory implementation on the GPU
    std::cout << "Running coalesced pinned implementation" << std::endl;
    coalescedConstitutive<<<gridSize, blockSize>>>(Fg, lambda, mu, Pg, nEval);

    // Wait for GPU to finish before accessing on host
    status = cudaDeviceSynchronize();
    checkError(status, "cudaDeviceSynchronize", __FILE__, __LINE__);

    // Check timing of executing computation
    status = cudaEventRecord(stopComp);
    checkError(status, "cudaEventRecord", __FILE__, __LINE__);
    status = cudaEventSynchronize(stopComp);
    checkError(status, "cudaEventSynchronize", __FILE__, __LINE__);

    milliseconds = 0.;
    status = cudaEventElapsedTime(&milliseconds, startComp, stopComp);
    checkError(status, "cudaEventElapsedTime", __FILE__, __LINE__);

    std::cout << "\nExecuting computation took: " << milliseconds << " ms" << std::endl
              << "Finished synchronizing device" << std::endl;

    // Receiving memory and timing the transaction
    status = cudaEventRecord(startSend);
    checkError(status, "cudaEventRecord", __FILE__, __LINE__);

    cudaMemcpy(P, Pg, nEval * 9 * sizeof(real), cudaMemcpyDeviceToHost);

    status = cudaEventRecord(stopSend);
    checkError(status, "cudaEventRecord", __FILE__, __LINE__);
    status = cudaEventSynchronize(stopSend);
    checkError(status, "cudaEventSynchronize", __FILE__, __LINE__);

    // Check the timing
    milliseconds = 0.;
    status = cudaEventElapsedTime(&milliseconds, startSend, stopSend);
    checkError(status, "cudaEventElapsedTime", __FILE__, __LINE__);

    std::cout << "\nReceiving memory took: " << milliseconds << " ms" << std::endl
              << "\n---------- Cuda Stop Timing ----------" << std::endl;

    // Destroy events
    status = cudaEventDestroy(startComp);
    checkError(status, "cudaEventDestroy", __FILE__, __LINE__);
    status = cudaEventDestroy(stopComp);
    checkError(status, "cudaEventDestroy", __FILE__, __LINE__);
    status = cudaEventDestroy(startSend);
    checkError(status, "cudaEventDestroy", __FILE__, __LINE__);
    status = cudaEventDestroy(stopSend);
    checkError(status, "cudaEventDestroy", __FILE__, __LINE__);

    // Verify each result is at the correct index
    std::cout << "\nStarting verification by index..." << std::endl;
    verifyCoalescedResults(nEval, lambda, mu, F, P, print);
    std::cout << "Finished verification by index..." << std::endl;

    // Free the allocated memory
    status = cudaFreeHost(F);
    checkError(status, "cudaFreeHost", __FILE__, __LINE__);
    status = cudaFreeHost(P);
    checkError(status, "cudaFreeHost", __FILE__, __LINE__);
    status = cudaFree(Fg);
    checkError(status, "cudaFree", __FILE__, __LINE__);
    status = cudaFree(Pg);
    checkError(status, "cudaFree", __FILE__, __LINE__);

    status = cudaDeviceReset();
    checkError(status, "cudaDeviceReset", __FILE__, __LINE__);

    return 0;
}
