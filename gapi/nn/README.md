# About Examples

The following examples are python files. Below includes a short description of
the example as well as the command(s) to run the program. These are examples that
use Groq API Neural Net Components

1. `matmul_fp16_compile.py` :: A standalone matrix multiplication python module in
Groq API using NN MatMul component with FP16 input tensors. This script only compiles
the matrix multiplication example. Use the runtime script to execute the generated IOP
file on Groq hardware.

    > python matmul_fp16_compile.py
    > python matmul_fp16_compile.py --help

2. `matmul_fp16_runtime.py` :: A standalone matrix multiplication python module in
Groq API using NN MatMul component with FP16 input tensors. This script only runs
the matrix multiplication example. Use the compile script to generate the IOP file.

    > python matmul_fp16_runtime.py
    > python matmul_fp16_runtime.py --help

3. `syrk.py` :: An example of using the nn SYRK component. This example uses
default values of shape (outer,inner) = (100,100), alpha = 2.3, gamma = 1.5
Users could change these parameters by running the command-line interface as:

    > python syrk.py --shape 10 50 --alpha 1.1 --gamma 2.1

4. `unpack_multibyte_tensor.py` :: An example that sets the input tensor
format to "CONTIGUOUS" to instruct the host runtime to serialize multibyte
values into the vector's inner dimension. The program's result tensor is
in the default "STRIDED" format to instruct the host runtime to transpose
the multibyte values back into a serialized format.

    > python unpack_multibyte.py

5. `gemm.py` :: This example uses default values of shape
(outer,inner) = (100,100), alpha = 2.3, gamma = 1.5
Users could change these parameters by running the command-line interface

    > python gemm.py --shape 10 50 --alpha 1.1 --gamma 2.1

6. `symm.py` :: This example uses default values of shape
(outer,inner) = (100,100), alpha = 2.3, gamma = 1.5
Users could change these parameters by running the command-line interface

    > python symm.py --shape 10 50 --alpha 1.1 --gamma 2.1
