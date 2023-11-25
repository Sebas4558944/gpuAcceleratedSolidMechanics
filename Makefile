# target: dependencies
#	rule:

# Define reusable commands across makefiles
# Set the compilers
NVCC = /usr/bin/nvcc
GCC = g++
LINK = /usr/lib/nvidia-cuda-toolkit/bin/g++
PROFILE = nsys profile
PROCESS = nsys stats

# Define flags for different compilations
PYRE_INCL = -I/home/sebas/dv/aces/pyre/build/lib

NVCC_FLAGS = -Xptxas -O3,-v -O3 -c

CC_FLAGS = -c -O3 

GCC_FLAGS = -pg -std=c++17

LINK_FLAGS = -lcudart_static -lrt -lpthread -ldl #$(PYRE_LINK)

PROFILE_FLAGS = --backtrace=dwarf --cuda-memory-usage=true --force-overwrite=true \
 				--gpu-metrics-device=all --gpuctxsw=true --trace=cuda,osrt

PROCESS_FLAGS = --report gputrace,cudaapisum --force-overwrite true --format csv \
				--timeunit milliseconds

# Set the size for execution
SIZE = 20

# For ease of use, make only one lib for all implementations
HOSTS = gradients verification
KERNELS = basicKernel symmetryKernel componentKernel axialKernel
DEVS = errors $(KERNELS) 
LIB = gpuLib

# Standard rule to compile exe defined above
.PHONY: all
all: $(EXE) 

# General rules to create and cleanup the folders
.PHONY: create
create:
	mkdir build/libs build/outputs build/archive

.PHONY: clean
clean: 
	rm build/archive/*.o

# Compile CPU code
.PHONY: gradients
gradients: helpers/gradients.cc
	$(GCC) $(CC_FLAGS) $(PYRE_INCL) $< -o build/archive/$@.o

.PHONY: verification
verification: helpers/verifyResults.cc
	$(GCC) $(CC_FLAGS) $(PYRE_INCL) $< -o build/archive/$@.o

# Compile GPU code
.PHONY: errors
errors: helpers/errorCheck.cu
	$(NVCC) $(NVCC_FLAGS) $< -o build/archive/$@.o

.PHONY: basicKernel
basicKernel: gpuKernels/basicKernels.cu
	$(NVCC) $(NVCC_FLAGS) $< -o build/archive/$@.o

.PHONY: axialKernel
axialKernel: gpuKernels/axialKernels.cu
	$(NVCC) $(NVCC_FLAGS) $< -o build/archive/$@.o

.PHONY: symmetryKernel
symmetryKernel: gpuKernels/symmetryKernels.cu
	$(NVCC) $(NVCC_FLAGS) $< -o build/archive/$@.o

.PHONY: componentKernel
componentKernel: gpuKernels/componentKernels.cu
	$(NVCC) $(NVCC_FLAGS) $< -o build/archive/$@.o

# Archive the output files together
.PHONY: $(LIB)
$(LIB): $(HOSTS) $(DEVS)
	ar rvs build/libs/$@.a build/archive/*.o 

# Compile the main files
.PHONY: benchmark
benchmark: implementations/benchmark.cu 
	$(NVCC) $(NVCC_FLAGS) $< -o build/outputs/$@.o

.PHONY: pinnend
pinnend: implementations/pinned.cu 
	$(NVCC) $(NVCC_FLAGS) $< -o build/outputs/$@.o

.PHONY: saturation
saturation: implementations/saturation.cu 
	$(NVCC) $(NVCC_FLAGS) $< -o build/outputs/$@.o

.PHONY: symmetry
symmetry: implementations/stressSymmetry.cu 
	$(NVCC) $(NVCC_FLAGS) $< -o build/outputs/$@.o

# Compile the executable
.PHONY: makeBenchmark
makeBenchmark: benchmark $(LIB)
	$(LINK) $(GCC_FLAGS) build/outputs/$<.o $(LINK_FLAGS) build/libs/$(LIB).a -o build/$@

.PHONY: makePinned
makePinned: pinnend $(LIB)
	$(LINK) $(GCC_FLAGS) build/outputs/$<.o $(LINK_FLAGS) build/libs/$(LIB).a -o build/$@

.PHONY: makeSaturation
makeSaturation: saturation $(LIB)
	$(LINK) $(GCC_FLAGS) build/outputs/$<.o $(LINK_FLAGS) build/libs/$(LIB).a -o build/$@

.PHONY: makeSymmetry
makeSymmetry: symmetry $(LIB)
	$(LINK) $(GCC_FLAGS) build/outputs/$<.o $(LINK_FLAGS) build/libs/$(LIB).a -o build/$@


# Commands to run different implementations
.PHONY: runBenchmark
runBenchmark: makeBenchmark
	build/makeBenchmark $(SIZE)

.PHONY: runPinned
runPinned: makePinned
	build/makePinned $(SIZE)

.PHONY: runSaturation
runSaturation: makeSaturation
	build/makeSaturation $(SIZE)

.PHONY: runSymmetry
runSymmetry: makeSymmetry
	build/makeSymmetry $(SIZE)
