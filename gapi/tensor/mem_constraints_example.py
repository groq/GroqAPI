"""Memory Constraints Example
This is an example on how/when to use the memory constraints that dictate a
relationship between the memory allocated to memory tensors.
"""
import os
import numpy as np

import groq.api as g
from groq.api import nn
from groq.runner import tsp


def compile(shape):
    # Defines a program that performs a*b + c

    # Declare input tensors.
    a = g.input_tensor(shape, g.float16, name="A")
    b = g.input_tensor(shape, g.float16, name="B", layout="H1(E), -1, S17")
    c = g.input_tensor(shape, g.float32, name="C")
    # By default, the memory allocated to a, b and c will all be slice exclusive.
    # However, in this example we know that they can all share the same slices as they
    # are accessed at different time. So we can override the default behavior and ask the
    # allocator to not necessarily make them slice exclusive.
    g.add_mem_constraints(
        [a, b, c], [a, b, c], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE
    )
    # Create the Transpose component and mark it as buffered.
    trans_comp = nn.TransposeMatrix(
        planes=[0, 1], skip_mem_copy=True, is_buffered=True, arith_mode_warmup=True
    )
    # Perform the transpose operation.
    b_t_mt = trans_comp(b, time=0)

    # Declare the MatMul component, mark it as buffered and set its
    # "predecessor" to be the Transpose comp.
    mm = nn.MatMul(
        planes=[0, 1],
        use_vxm_accum=True,
        buffer_output=True,
        is_buffered=True,
        arith_mode_warmup=True,
    )

    # Build matmul
    ab_mt = mm(a, b_t_mt, predecessors=[trans_comp])

    # Define a buffered resource scope and set its
    # "predecessor to be the MatMul component.
    with g.ResourceScope(
        name="add_stage", time=None, is_buffered=True, predecessors=[mm]
    ):
        add_res_t = ab_mt.add(c, alus=[0], time=0)
        add_res_mt = add_res_t.write(name="final_result")

    iop_file = g.compile(
        base_name="mul_add", result_tensor=add_res_mt, gen_vis_data=True
    )
    return iop_file


def run(iop_file, shape):

    np.set_printoptions(linewidth=1000, threshold=10000)

    if not os.path.exists(iop_file):
        raise Exception(f"IOP file does not exist: {iop_file}")

    print(f"Running programs from {iop_file}")

    a_data = (np.random.rand(*shape)).astype(np.float16)
    b_data = (np.random.rand(*shape)).astype(np.float16)
    c_data = (np.random.rand(*shape)).astype(np.float32)
    mac_program = tsp.create_tsp_runner(iop_file)

    result = mac_program(A=a_data, B=b_data, C=c_data)

    ab_oracle = np.matmul(a_data, b_data, dtype=np.float32)
    oracle = np.add(ab_oracle, c_data)
    np.testing.assert_allclose(result["final_result"], oracle, atol=0.01)


def main():
    """Compiles and runs the example program."""
    shape = (320, 320)

    iop_file = compile(shape)
    print(f"Program compiled to IOP file: {iop_file}")
    run(iop_file, shape)


if __name__ == "__main__":
    main()
