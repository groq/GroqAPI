# About
`matmul_example.cpp` is an example program written in C++ that performs an INT8
matrix multiplication on both the GroqChip and the host CPU (the latter computation 
is implemented naively and is intended only to verify the computation performed by 
the GroqChip).

# Prerequisites

1. Installation of the latest GroqWare Suite SDK
2. A C++ compiler
3. CMake (can be installed by running `sudo apt install build-essential cmake` post-SDK installation)

# Build Instructions

1. Run `mkdir build`
2. Navigate to the newly created `build` directory by running `cd build`
3. Run `cmake ..`
4. Run `make ./matmul_example ../mm_int8_100by1000_x_400by1000.iop` - the
expected output should be "OK" (see the code for details)!
