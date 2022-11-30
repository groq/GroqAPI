# Copyright (c) 2020-2021, Groq Inc. All rights reserved.
# Groq, Inc. and its licensors retain all intellectual property and proprietary
# rights in and to this software, related documentation and any modifications
# thereto. Any use, reproduction, modification, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Groq, Inc. is strictly prohibited.

"""
Example script for a matmul tensor op.
This example uses constants as inputs and writes to program output.
"""
# pylint: disable=no-member
# pylint: disable=import-error
import numpy as np
from groq.runner import tsp
import groq.api as g


def build_matmul(inp1, inp2):
    # Create 2 input tensors as constants initialized with with random data.
    # IMPL_DETAIL: setting allocation request to force single hemisphere and slice concurrency.
    t1 = g.from_data(data=inp1, name="A", layout="H1(W), -1, S1")
    t2 = g.from_data(data=inp2, name="B", layout="H1(W), -1, S16")

    # Clear the MXM planes to prep for any matrix multiplications
    g.clear_mxm(
        planes=[0, 1, 2, 3],
        time=-22,  # 20 cycles to clear the MXM, 2 cycles to load the data
        dtype=t1.dtype,  # set the data type to match the input data
    )

    # Use "matmul" method on the input tensor.
    result_st = t1.matmul(t2, planes=[0], time=20)
    # OR use "matmul" function.
    # result_st = g.matmul(t1, t2, planes=[0], time=20)
    # OR use "nn.MatMul" component.
    # mm = nn.MatMul(planes=[0], time=20)
    # result_st = mm(t1, t2)
    # ^^ On all above ops, need to select the mxm plane (single plane for int8) and set
    # time on the matmul operation
    # Write tensor to memory.
    result_mt = result_st.write(
        name="mm_result", program_output=True, layout="H1(W), -1, S4"
    )
    return result_mt


if __name__ == "__main__":
    # IMP DETAILS on using tensor matmul function:
    # - Expects both inputs as rank-2 tensor.
    # - The inner dimension on both tensors should match and cannot exceed 320.
    # - The outer dimension for the second tensor (inp2) should not exceed 320 and
    #   should be a multiple of 16.
    # - You can use nn.MatMul component which doesn't have any restriction on inner or
    #   outer sizes.
    inp1 = np.random.randint(-128, 127, (100, 320), np.int8)
    inp2 = np.random.randint(-128, 127, (160, 320), np.int8)
    print(
        "\nBuilding INT8 matmul for input tensors {} x {}".format(
            inp1.shape, inp2.shape
        )
    )
    result_mt = build_matmul(inp1, inp2)
    print("\nCompiling and Running on HW...")

    # Create an IOP file using Groq API compile function
    iop_file = g.compile(base_name="matmul", result_tensor=result_mt, gen_vis_data=True)
    # Create MatMul program instance by calling TSP runner and passing in IOP file to run
    matmul_program = tsp.create_tsp_runner(iop_file)
    actual = matmul_program()
    matmul_result = actual["mm_result"]

    oracle = np.matmul(inp1, inp2.transpose(), dtype=np.int32)
    max_atol = max(abs(oracle.reshape(-1) - matmul_result.reshape(-1)))

    if max_atol <= 0.00001:
        print(f"\nTest PASSED with a max tolerance of {max_atol}")
    else:
        print(
            f"\nTest FAILED with a max tolerance of {max_atol} (should be <= 0.00001)"
        )
