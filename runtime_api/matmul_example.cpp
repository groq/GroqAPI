/**
 * This program expects an IOP file produced by the following Python code, which was
 * inspired by the `matmul_fp16.py` example. The code below differs from `matmul_fp16.py`
 * in that it uses INT8 rather than FLOAT16.
 *
 *  import groq.api as g
 *  import groq.api.nn as nn
 *
 *  # Create 2 input tensors.
 *  t1 = g.input_tensor(shape=(100, 1000), dtype=g.int8, name="A")
 *  t2 = g.input_tensor(shape=(400, 1000), dtype=g.int8, name="B")
 *
 *  # Instantiate MatMul component.
 *  mm = nn.MatMul(time=20, buffer_output=True)
 *
 *  # Build MatMul component.
 *  result_mt = mm(t1, t2)
 *
 *  # Mark `result_mt` as program output.
 *  result_mt.set_program_output()
 *
 *  # Compile program to generate IOP file.
 *  iop_file = g.compile(base_name="mm_example", gen_vis_data=True, check_stream_conflicts=True)
 */

#include "Device.hpp"
#include "Driver.hpp"
#include "IOP.hpp"
#include "Misc.hpp"
#include "SimpleRunner.hpp"

#include <algorithm>
#include <iostream>
#include <random>

/**
 * @brief This class represents a two-dimensional matrix. Instances of this class are used
 * to hold the two input matrices - the output matrix (from GroqChip), and an oracle
 * matrix (computed by the mult method from the CPU and used to verify the result from the GroqChip).
 */
template <typename T>
struct SimpleMatrix
{
    // To hold the output matrix from GroqChip
    SimpleMatrix(size_t rows, size_t cols)
        : nrows(rows)
        , ncols(cols)
        , data(nrows * ncols, 0)
    {
    }

    const T &at(size_t row, size_t col) const
    {
        if (row >= nrows)
            throw std::out_of_range("bad row");
        if (col >= ncols)
            throw std::out_of_range("bad col");
        return data.at((row * ncols) + col);
    }

    T &at(size_t row, size_t col)
    {
        if (row >= nrows)
            throw std::out_of_range("bad row");
        if (col >= ncols)
            throw std::out_of_range("bad col");
        return data.at((row * ncols) + col);
    }

    // To hold the oracle matrix from the CPU
    template <typename R>
    SimpleMatrix<R> mult(const SimpleMatrix<T> &other) const
    {
        SimpleMatrix<R> result(nrows, other.ncols);
        for (size_t i = 0; i < nrows; ++i) {
            for (size_t j = 0; j < other.ncols; ++j) {
                for (size_t k = 0; k < ncols; ++k) {
                    R t1 = at(i, k);
                    R t2 = other.at(k, j);
                    result.at(i, j) += (t1 * t2);
                }
            }
        }
        return result;
    }

    SimpleMatrix<T> transpose() const
    {
        SimpleMatrix<T> other = *this;
        size_t n = 0;

        for (size_t j = 0; j < ncols; ++j) {
            for (size_t i = 0; i < nrows; ++i) {
                other.data[n++] = this->at(i, j);
            }
        }
        other.nrows = ncols;
        other.ncols = nrows;
        return other;
    }

    uint8_t *raw()
    {
        return reinterpret_cast<uint8_t *>(&data[0]);
    }

    size_t rawSize() const
    {
        return data.size() * sizeof(T);
    }

    bool operator==(const SimpleMatrix<T> &other) const
    {
        return data == other.data && nrows == other.nrows && ncols == other.ncols;
    }

    size_t nrows;
    size_t ncols;
    std::vector<T> data;
};

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cout << "Usage\n" << argv[0] << " </path/to/mm_int8_100by1000_x_400by1000.iop>" << std::endl;
        std::exit(1);
    }

    // Read IOP data from given IOP file
    auto iopData = readFile(argv[1]);
    if (iopData.empty()) {
        std::cout << "Invalid IOP file" << std::endl;
        std::exit(1);
    }

    // Create two input matrices named `a` and `b`
    SimpleMatrix<int8_t> a(100, 1000);
    SimpleMatrix<int8_t> b(400, 1000);

    // Fill matrices a and b with randomly generated numbers
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_int_distribution<int8_t> dist;
    std::generate(a.data.begin(), a.data.end(), [&] { return dist(eng); });
    std::generate(b.data.begin(), b.data.end(), [&] { return dist(eng); });

    // Generate oracle for comparison with what GroqChip computes
    SimpleMatrix<int32_t> oracle = a.mult<int32_t>(b.transpose());

    // Access GroqCard
    groq::Driver driver;
    groq::Device device = driver.getNextDevice();
    device.open();
    device.reset();
    device.clearMemory();

    // Load IOP file data
    const groq::IOP iop(&iopData[0], iopData.size());
    device.loadProgram(iop, 0, false);

    // Create runner
    groq::SimpleRunner runner(driver, iop);

    // Create result buffer
    SimpleMatrix<int32_t> result(oracle.nrows, oracle.ncols);

    // Load inputs into runner
    runner.addInputBuffer(a.raw(), a.rawSize(), 1);
    runner.addInputBuffer(b.raw(), b.rawSize(), 0);

    // Tell runner where result should go
    runner.addOutputBuffer(result.raw(), result.rawSize(), 0);

    // Perform MatMul on GroqChip
    runner.invoke(device);

    // Print result of check against oracle
    std::cout << (oracle == result ? "OK" : "FAIL") << std::endl;

    return 0;
    unused(argc);
    unused(argv);

    // The following is for Python users who want to check the results:
#if 0
    // Write the data of a, b, the oracle, and GroqChip results to .bin files
    writeFile("a.bin", a.raw(), a.rawSize());
    writeFile("b.bin", b.raw(), b.rawSize());
    writeFile("oracle.bin", oracle.raw(), oracle.rawSize());
    writeFile("result.bin", result.raw(), result.rawSize());
#endif

    // Python code to process a.bin, b.bin, oracle.bin, result.bin
#if 0
    import numpy as np

    a = np.fromfile('a.bin', dtype=np.int8).reshape((100,1000))
    b = np.fromfile('b.bin', dtype=np.int8).reshape((400,1000))
    oracle = np.fromfile('oracle.bin', dtype=np.int32).reshape((100,400))
    result = np.fromfile('result.bin', dtype=np.int32).reshape((100,400))

    if not np.array_equal(oracle, result):
        print("fail")
    if not np.array_equal(result, np.matmul(a, b.transpose(), dtype=np.int32)):
        print("fail")
#endif
}
