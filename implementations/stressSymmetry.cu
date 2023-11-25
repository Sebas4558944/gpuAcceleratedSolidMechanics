// Include the header files needed
#include <iostream>

#include "../gpuKernels/symmetryKernels.h"
#include "../gpuKernels/axialKernels.h"
#include "../helpers/errorCheck.h"
#include "../helpers/verifyResults.h"
#include "../helpers/gradients.h"

int main(int argc, char **argv)
{
    // Initialize error status
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
    int nAlloc = nEval * 9;

    // Log how many iterations are performed
    std::cout << "\n---------- Launch Configuration ----------" << std::endl
              << "Number of quadrature points:      " << quads << std::endl
              << "Number of elements :              " << elem / 1e6 << " M " << std::endl
              << "Number of constitutive updates:   " << nEval / 1e6 << " M" << std::endl;

    // Allocate threads for each block
    int nThreadsPerBlock = 512;

    // Set the gridsize (total number of threads) optimal amount varies per problem
    int gridsize = 400000;

    int nBlocks, nThreadsLastBlock, nLoops;
    int nThreadsLastLoop, nBlocksLastLoop;
    if (nEval > nThreadsPerBlock)
    {
        // Retrieve the properties of the GPU
        cudaDeviceProp prop;
        status = cudaGetDeviceProperties(&prop, 0);
        checkError(status, "cudaGetDeviceProperties", __FILE__, __LINE__);

        // Determine the instruction size (= amount of threads executed in parallel by all SM's)
        // There are 32 threads per SM
        int instructionSize = prop.multiProcessorCount * 32;

        // Determine the number of blocks
        nBlocks = gridsize / nThreadsPerBlock + 1;

        // Check if the last block is being used
        int nThreadsLastBlock = gridsize - (nBlocks - 1) * nThreadsPerBlock;
        if (nThreadsLastBlock == 0)
        {
            nBlocks -= 1;
            nThreadsLastBlock = nThreadsPerBlock;
        }

        // Determine the amount of loops over the saturation size to complete the problem size
        nLoops = nEval / gridsize + 1;

        // Determine the amount of blocks in the last loop
        nThreadsLastLoop = nEval - (nLoops - 1) * gridsize;

        // Check if the last loop is necessary
        int nThreadsLastLoopBlock;
        if (nThreadsLastLoop == 0)
        {
            nLoops -= 1;
            nBlocksLastLoop = nBlocks;
            nThreadsLastLoopBlock = nThreadsPerBlock;
        }
        else
        {
            // Check if the last block in the last loop is being used
            nBlocksLastLoop = nThreadsLastLoop / nThreadsPerBlock + 1;
            nThreadsLastLoopBlock = nThreadsLastLoop - (nBlocksLastLoop - 1) * nThreadsPerBlock;
            if (nThreadsLastLoopBlock == 0)
            {
                nBlocksLastLoop -= 1;
                nThreadsLastLoopBlock = nThreadsPerBlock;
            }
        }

        std::cout << "\nInstruction size:                 " << instructionSize << std::endl
                  << "Used saturation amount:           " << gridsize << std::endl
                  << "Number of rounds:                 " << nLoops << std::endl
                  << "Number of blocks:                 " << nBlocks << std::endl
                  << "Number of threads:                " << nBlocks * nThreadsPerBlock << std::endl
                  << "Last block:                       " << nThreadsLastBlock << std::endl
                  << "Active blocks in last round:      " << nBlocksLastLoop << std::endl
                  << "Active threads in last round:     " << nBlocksLastLoop * nThreadsPerBlock << std::endl
                  << "Active threads in last block:     " << nThreadsLastLoopBlock << std::endl;
    }
    else
    {
        nThreadsPerBlock = nEval;
        nBlocks = 1;
        nThreadsLastBlock = nThreadsPerBlock;
    }

    // Create cuda timers
    cudaEvent_t startComp, stopComp;
    status = cudaEventCreate(&startComp);
    checkError(status, "cudaEventCreate", __FILE__, __LINE__);
    status = cudaEventCreate(&stopComp);
    checkError(status, "cudaEventCreate", __FILE__, __LINE__);

    // Initialize variables to be stored in the cpu
    real *F, *P;
    status = cudaMallocHost(&F, nAlloc * sizeof(real));
    checkError(status, "cudaMallocHost", __FILE__, __LINE__);

    status = cudaMallocHost(&P, nAlloc * sizeof(real));
    checkError(status, "cudaMallocHost", __FILE__, __LINE__);

    // Define an an array of random deformation gradients as a perturbation of the identity
    real F_pert = 0.1;
    getCoalescedGradients(F_pert, nEval, F, print);

    // Allocate memory on the GPU
    real *Fg, *Pg;
    status = cudaMalloc(&Fg, nAlloc * sizeof(real));
    checkError(status, "cudaMalloc", __FILE__, __LINE__);

    status = cudaMalloc(&Pg, nAlloc * sizeof(real));
    checkError(status, "cudaMalloc", __FILE__, __LINE__);

    // Create cuda streams
    cudaStream_t *streams;
    cudaMalloc(&streams, 3 * nLoops * sizeof(cudaStream_t));
    for (int i = 0; i < (3 * nLoops); i++)
    {
        status = cudaStreamCreate(&(streams[i]));
        checkError(status, "cudaStreamCreate", __FILE__, __LINE__);
    }

    std::cout << "\n---------- Cuda Event Timing ----------" << std::endl;

    // Executing computation
    status = cudaEventRecord(startComp);
    checkError(status, "cudaEventRecord", __FILE__, __LINE__);
    std::cout << "\nSynchronizing device..." << std::endl;

    // Compute the diagonal components
    int loopSize = gridsize;
    int loopBlocks = nBlocks;
    for (int i = 0; i < nLoops; i++)
    {
        // Set the size for the last loop
        if (i == (nLoops - 1))
        {
            loopSize = nThreadsLastLoop;
            loopBlocks = nBlocksLastLoop;
        }
        else
        {
            loopSize = gridsize;
            loopBlocks = nBlocks;
        }

        // Send input data for the saturated computation
        status = cudaMemcpyAsync(Fg + i * gridsize, F + i * gridsize, loopSize * sizeof(real), cudaMemcpyHostToDevice, streams[i * 3]);
        checkError(status, "cudaMemcpyAsync", __FILE__, __LINE__);

        status = cudaMemcpyAsync(Fg + i * gridsize + 4 * nEval, F + i * gridsize + 4 * nEval, loopSize * sizeof(real), cudaMemcpyHostToDevice, streams[i * 3 + 1]);
        checkError(status, "cudaMemcpyAsync", __FILE__, __LINE__);

        status = cudaMemcpyAsync(Fg + i * gridsize + 8 * nEval, F + i * gridsize + 8 * nEval, loopSize * sizeof(real), cudaMemcpyHostToDevice, streams[i * 3 + 2]);
        checkError(status, "cudaMemcpyAsync", __FILE__, __LINE__);

        // Synchronize the streams
        cudaStreamSynchronize(streams[i * 3]);
        cudaStreamSynchronize(streams[i * 3 + 1]);
        cudaStreamSynchronize(streams[i * 3 + 2]);

        // Perform the saturated computation of P11
        axial11<<<loopBlocks, nThreadsPerBlock, 0, streams[i * 3]>>>(Fg + i * gridsize, lambda, mu, Pg + i * gridsize, nEval, loopSize);

        // Receive results from the saturated computation
        status = cudaMemcpyAsync(P + i * gridsize, Pg + i * gridsize, loopSize * sizeof(real), cudaMemcpyDeviceToHost, streams[i * 3]);
        checkError(status, "cudaMemcpyAsync", __FILE__, __LINE__);

        // Perform the saturated computation of P22
        axial22<<<loopBlocks, nThreadsPerBlock, 0, streams[i * 3 + 1]>>>(Fg + i * gridsize, lambda, mu, Pg + i * gridsize + 4 * nEval, nEval, loopSize);

        status = cudaMemcpyAsync(P + i * gridsize + 4 * nEval, Pg + i * gridsize + 4 * nEval, loopSize * sizeof(real), cudaMemcpyDeviceToHost, streams[i * 3 + 1]);
        checkError(status, "cudaMemcpyAsync", __FILE__, __LINE__);

        // Perform the saturated computation of P22
        axial33<<<loopBlocks, nThreadsPerBlock, 0, streams[i * 3 + 2]>>>(Fg + i * gridsize, lambda, mu, Pg + i * gridsize + 8 * nEval, nEval, loopSize);

        status = cudaMemcpyAsync(P + i * gridsize + 8 * nEval, Pg + i * gridsize + 8 * nEval, loopSize * sizeof(real), cudaMemcpyDeviceToHost, streams[i * 3 + 2]);
        checkError(status, "cudaMemcpyAsync", __FILE__, __LINE__);
    }

    // Compute the shear components
    int iShear[3] = {1, 2, 5};
    int jShear[3] = {3, 6, 7};
    int iComponent, jComponent;
    for (int shearComp = 0; shearComp < 3; shearComp++)
    {
        // Set the indices of the current shear components
        iComponent = iShear[shearComp];
        jComponent = jShear[shearComp];
        for (int i = 0; i < nLoops; i++)
        {
            // Set the size for the last loop
            if (i == (nLoops - 1))
            {
                loopSize = nThreadsLastLoop;
                loopBlocks = nBlocksLastLoop;
            }
            else
            {
                loopSize = gridsize;
                loopBlocks = nBlocks;
            }

            // Send input data for the saturated computation
            status = cudaMemcpyAsync(Fg + i * gridsize + iComponent * nEval, F + i * gridsize + iComponent * nEval, loopSize * sizeof(real), cudaMemcpyHostToDevice, streams[i]);
            checkError(status, "cudaMemcpyAsync", __FILE__, __LINE__);

            status = cudaMemcpyAsync(Fg + i * gridsize + jComponent * nEval, F + i * gridsize + jComponent * nEval, loopSize * sizeof(real), cudaMemcpyHostToDevice, streams[i]);
            checkError(status, "cudaMemcpyAsync", __FILE__, __LINE__);

            // Perform the saturated computation
            shear<<<loopBlocks, nThreadsPerBlock, 0, streams[i]>>>(Fg + i * gridsize, lambda, mu, Pg + i * gridsize, nEval, iComponent, jComponent, loopSize);

            // Receive results from the saturated computation
            status = cudaMemcpyAsync(P + i * gridsize + iComponent * nEval, Pg + i * gridsize + iComponent * nEval, loopSize * sizeof(real), cudaMemcpyDeviceToHost, streams[i]);
            checkError(status, "cudaMemcpyAsync", __FILE__, __LINE__);
        }
    }

    // Wait for GPU to finish before accessing on host
    status = cudaDeviceSynchronize();
    checkError(status, "cudaDeviceSynchronize", __FILE__, __LINE__);

    // Check timing of executing computation
    status = cudaEventRecord(stopComp);
    checkError(status, "cudaEventRecord", __FILE__, __LINE__);
    status = cudaEventSynchronize(stopComp);
    checkError(status, "cudaEventSynchronize", __FILE__, __LINE__);

    // Compute the elapsed time
    float milliseconds = 0.;
    status = cudaEventElapsedTime(&milliseconds, startComp, stopComp);
    checkError(status, "cudaEventElapsedTime", __FILE__, __LINE__);

    std::cout << "Finished synchronizing device" << std::endl
              << "\nExecuting computation took: " << milliseconds << " ms" << std::endl;

    std::cout << "\n---------- Cuda Stop Timing ----------" << std::endl;

    // Destroy events
    status = cudaEventDestroy(startComp);
    checkError(status, "cudaEventDestroy", __FILE__, __LINE__);
    status = cudaEventDestroy(stopComp);
    checkError(status, "cudaEventDestroy", __FILE__, __LINE__);

    // Destroy the cuda streams
    for (int i = 0; i < (3 * nLoops); i++)
    {
        status = cudaStreamDestroy(streams[i]);
        checkError(status, "cudaStreamDestroy", __FILE__, __LINE__);
    }

    // Verify each result is at the correct index
    std::cout << "\nStarting verification by index..." << std::endl;
    verifyCoalescedResults(nEval, lambda, mu, F, P, print);
    std::cout << "Finished verification by index..." << std::endl;

    // Free the allocated memory
    status = cudaFree(Fg);
    checkError(status, "cudaFree", __FILE__, __LINE__);

    status = cudaFree(Pg);
    checkError(status, "cudaFree", __FILE__, __LINE__);

    status = cudaFreeHost(F);
    checkError(status, "cudaFreeHost", __FILE__, __LINE__);

    status = cudaFreeHost(P);
    checkError(status, "cudaFreeHost", __FILE__, __LINE__);

    // Reset the device in the end
    status = cudaDeviceReset();
    checkError(status, "cudaDeviceReset", __FILE__, __LINE__);

    return 0;
}