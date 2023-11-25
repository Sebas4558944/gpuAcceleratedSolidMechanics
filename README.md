## Intro
This repository contains both the code and slides of the presentation on GPU-Accelerated Finite Element Analysis 
for Solid Mechanics given in July at Delft University of Technology. Below is a short description of this repository

## Makefile
Using the Makefile, one can compile and run different GPU implementations of a simple FEM solver.
The difference in time each implementation takes can be reviewed by the results presented in the terminal.
It is recommended to investigate the results deeper using NVIDIA tools to analyze GPU-code performance.
Please visit my thesis (available through the TU Delft repository) or the NVIDIA website for more in-depth documentation on this.

## gpuKernels
Contains the different GPU kernels used in the GPU implementations.

## helpers
Contains helper functions to compute gradients, check for errors and verify results of the GPU implementation.

## implementations
Contains different GPU implementations based on the presented slides of the presentation.

If you have any questions, do not hesitate to reach out!
