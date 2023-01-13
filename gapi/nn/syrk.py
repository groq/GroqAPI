# Copyright (c) 2022, Groq Inc. All rights reserved.
#
# Groq, Inc. and its licensors retain all intellectual property and proprietary
# rights in and to this software, related documentation and any modifications
# thereto. Any use, reproduction, modification, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Groq, Inc. is strictly prohibited.

"""
Example script for using a nn SYRK component.

To run this example:
>> python syrk.py

This example uses default values of shape (outer,inner) = (100,100), alpha = 2.3, gamma = 1.5
Users could change these parameters by running the command-line interface
>> python syrk.py --shape 10 50 --alpha 1.1 --gamma 2.1

For more information on running this example, use:
>> python syrk.py --help
"""
# pylint: disable=no-member
# pylint: disable=import-error
import sys
import os
import argparse
import numpy as np
import groq.api as g
from groq.runner import tsp
from groq.api import nn
from groq.common import print_utils

helpers_dir = os.path.join(os.path.dirname(__file__), "../example_helpers")
sys.path.insert(0, helpers_dir)
import helpers


def compile_syrk(weight_shape, state_shape, alpha, gamma):
    """Compile a program that creates a SYRK component with the supplied data"""

    # Initialize input tensors
    weight_mt = g.input_tensor(shape=weight_shape, dtype=g.float16, name="weight_mt")
    state_mt = g.input_tensor(shape=state_shape, dtype=g.float32, name="state_mt")

    # Clear the MXM planes to prep for any matrix multiplications
    g.clear_mxm(
        planes=[0, 1, 2, 3],
        time=-22,
        dtype=g.float16,
    )

    # Instantiate the SYRK component
    syrk_model = nn.SYRK(alpha, gamma, time=0)
    result_mt = syrk_model(weight_mt, state_mt)

    print_utils.infoc("Compiling SYRK Example...")
    # Create IOP file of program
    iop_file = g.compile(
        base_name="syrk",
        result_tensor=result_mt,
        gen_vis_data=True,
        output_dir=os.getcwd(),
    )
    print_utils.infoc(f"\nModel successfully compiled and saved at {iop_file}")

    return


def runtime_syrk(wt_data, st_data, alpha, gamma):

    # Get IOP file of program
    iop_file = os.path.join(os.getcwd(), "syrk.iop")

    if os.path.exists(iop_file) is not True:
        print_utils.err("No IOP file found. Run the entire script using python syrk.py")
        return

    # Before running on hardware, check that hardware is available
    if helpers.get_num_chips_available() is None:
        print_utils.err(
            f"No Hardware Available. Copy {iop_file} to a system with GroqCard™ accelerators installed."
        )
        return

    # Run IOP file with the TSP runner function
    syrk_pgm = tsp.create_tsp_runner(iop_file)
    actual = syrk_pgm(weight_mt=wt_data, state_mt=st_data)
    syrk_result = actual["BLASL3_result"]

    # Compute Oracle.
    weights_np = np.reshape(wt_data, [outer, inner])
    scaled_state = gamma * np.reshape(st_data, [outer, outer])
    product = alpha * np.matmul(weights_np, np.transpose(weights_np), dtype=np.float32)
    oracle = np.reshape(np.add(scaled_state, product), [1, outer, outer])
    oracle = np.float32(oracle)

    print_utils.infoc(f"SYRK result is: {syrk_result}\n")
    print_utils.infoc(f"Numpy oracle result is: {oracle}\n")

    if np.allclose(syrk_result, oracle, atol=0.00001):
        print_utils.infoc("Groq's SYRK results match oracle results.")
    else:
        print_utils.infoc(f"Oracle shape={oracle.shape} data=\n{oracle}")
        print_utils.infoc(f"SYRK result shape={syrk_result.shape} data=\n{syrk_result}")
        print_utils.err(
            "Unexpected SYRK results. Check that your alpha and gamma values have not changed since you compiled.\n"
        )

    return (syrk_result, oracle)


if __name__ == "__main__":
    # Before getting started, check that the SDK is installed
    if helpers.find_sdk("groq-devtools") is None:
        raise Exception("No SDK Found")

    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="SYRK example")
    parser.add_argument(
        "--shape",
        default=[100, 100],
        type=int,
        nargs=2,
        help="Shape of the update matrix (before squaring). Default: 100,100",
    )
    parser.add_argument(
        "--alpha",
        default=[2.3],
        type=float,
        nargs=1,
        help="Scaling factor for the squared update matrix. Default: 2.3",
    )
    parser.add_argument(
        "--gamma",
        default=[1.5],
        type=float,
        nargs=1,
        help="Scaling factor for the state matrix. Default: 1.5",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use this flag to compile the example but not run on hardware.",
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Use this flag to run the previously compiled IOP file.",
    )
    args = parser.parse_args()

    alpha = args.alpha[0]
    gamma = args.gamma[0]
    outer = args.shape[0]
    inner = args.shape[1]
    compile = args.compile
    runtime = args.runtime

    # Generate random inputs and run the test
    print_utils.infoc(f"Using symmetric rank-k with shape: [{outer}, {inner}]")
    weights_shape = (outer, inner)
    state_shape = (outer, outer)

    # Create activation and weight tensors for FP16 syrk.
    wt_data = 10 * np.random.rand(*weights_shape).astype(np.float16).reshape(
        weights_shape
    )
    st_data = 100 * np.random.rand(*state_shape).astype(np.float32).reshape(state_shape)

    if not compile and not runtime:
        print_utils.infoc(
            "You have not specified whether you want to compile or run this example. Therefore, we'll assume you wanted to both compile and run this example."
        )
        compile = True
        runtime = True
    if compile:
        compile_syrk(weights_shape, state_shape, alpha, gamma)
    if runtime:
        # TODO: check that runtime input sizes match compiled sizes
        runtime_syrk(wt_data, st_data, alpha, gamma)
