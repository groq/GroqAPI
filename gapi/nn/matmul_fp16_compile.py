# Copyright (c) 2020-2022, Groq Inc. All rights reserved.
#
# Groq, Inc. and its licensors retain all intellectual property and proprietary
# rights in and to this software, related documentation and any modifications
# thereto.

"""Example script for using the Neural Network component: MatMul

This script compiles an IOP file that can be used to run on hardware using matmul_fp16_runtime.py.
To run this example:

$ python matmul_fp16_compile.py

This example uses input tensors as inputs and writes to program output.
"""
# pylint: disable=import-error
import groq.api as g
import groq.api.nn as nn
from groq.common import print_utils
import sys
import os

helpers_dir = os.path.join(os.path.dirname(__file__), "../example_helpers")
sys.path.insert(0, helpers_dir)
import helpers


def main():

    # Implementation details on using nn.Matmul component:
    # - Expects both inputs to be a rank-2 tensor.
    # - The inner dimension on both tensors should match.
    # - Note that the second tensor is implicitly transposed.

    # Create 2 input tensors.
    t1 = g.input_tensor(shape=(100, 1000), dtype=g.float16, name="A")
    t2 = g.input_tensor(shape=(400, 1000), dtype=g.float16, name="B")

    print_utils.infoc(
        f"\nBuilding FP16 matmul for input tensors of shapes {t1.shape} x {t2.shape}"
    )

    # Instantiate matmul component.
    mm = nn.MatMul(time=20, buffer_output=True, arith_mode_warmup=True)

    # ^^ Don't need to select any mxm plane or set the memory layouts.
    # Only thing you need to set is the time at which matmul should be scheduled.
    # Also you can pass buffer_output=True to avoid explicit write to memory.

    # Build matmul component.
    result_mt = mm(t1, t2)

    # Compile program to generate IOP. Also generate GroqView file.
    print_utils.infoc("\nCompiling model ...")
    iop_file = g.compile(
        base_name="mm_fp16",
        result_tensor=result_mt,
        gen_vis_data=True,
        output_dir=os.getcwd(),
    )
    print_utils.infoc(f"\nModel successfully compiled and saved at {iop_file}")


if __name__ == "__main__":
    # Before getting started, check that the SDK is installed
    if helpers.find_sdk("groq-devtools") is None:
        raise Exception("No SDK Found")
    else:
        main()
