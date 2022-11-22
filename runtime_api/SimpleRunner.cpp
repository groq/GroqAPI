#include "SimpleRunner.hpp"

#include "Device.hpp"
#include "Driver.hpp"
#include "IOP.hpp"
#include "Misc.hpp"

#include <cassert>

namespace groq {

SimpleRunner::SimpleRunner(Driver &driver, const IOP &iop, size_t programIndex, size_t entrypointIndex)
    : iop(iop)
    , programIndex(programIndex)
    , entrypointIndex(entrypointIndex)
    , tspInputSize(groq_program_get_input_size(iop.handle(), programIndex))
    , tspOutputSize(groq_program_get_output_size(iop.handle(), programIndex))
    , numInputs(inputTensorLayouts().size())
    , numOutputs(outputTensorLayouts().size())
    , inputIoba(nullptr)
    , outputIoba(nullptr)
    , inputBuffers(numInputs, nullptr)
    , outputBuffers(numOutputs, nullptr)
    , inputSizes(numInputs, 0)
    , outputSizes(numOutputs, 0)
{
    GROQOK(groq_allocate_inputs_iobuffer_array(driver.handle(), iop.handle(), 1, &inputIoba));
    GROQOK(groq_allocate_outputs_iobuffer_array(driver.handle(), iop.handle(), 1, &outputIoba));

    assert(inputIoba);
    assert(outputIoba);
}

SimpleRunner::~SimpleRunner()
{
    ::Status status;
    status = groq_deallocate_iobuffer_array(inputIoba);
    status = groq_deallocate_iobuffer_array(outputIoba);
    unused(status);
}

void SimpleRunner::addInputBuffer(uint8_t *buffer, size_t size, size_t index)
{
    const auto &layout = inputTensorLayouts().at(index);

    if (size != layout.getHostSize()) {
        throw std::runtime_error("Bad data size; expected " + std::to_string(layout.getHostSize()) + " got "
                                 + std::to_string(size));
    }

    assert(buffer);

    inputBuffers.at(index) = buffer;
    inputSizes.at(index) = size;
}

void SimpleRunner::addOutputBuffer(uint8_t *buffer, size_t size, size_t index)
{
    const auto &layout = outputTensorLayouts().at(index);

    if (size != layout.getHostSize()) {
        throw std::runtime_error("Bad data size; expected " + std::to_string(layout.getHostSize()) + " got "
                                 + std::to_string(size));
    }

    assert(buffer);

    outputBuffers.at(index) = buffer;
    outputSizes.at(index) = size;
}

void SimpleRunner::invoke(Device &device)
{
    assert(inputIoba);
    assert(outputIoba);

    // transform user's input data into layout expected by TSP
    for (size_t i = 0; i < numInputs; ++i) {
        const auto &layout = inputTensorLayouts().at(i);

        uint8_t *input = inputBuffers.at(i);
        uint8_t *output = nullptr;
        size_t inputSize = inputSizes.at(i);
        size_t outputSize = tspInputSize;

        GROQOK(groq_get_data_handle(inputIoba, 0, &output));

        assert(input);
        assert(output);

        layout.fromHost(input, inputSize, output, outputSize);
    }

    ::Completion completion;
    size_t nthInput = 0;
    size_t nthOutput = 0;

    GROQOK(groq_invoke(device.handle(), inputIoba, nthInput, outputIoba, nthOutput, &completion));
    GROQOK(groq_wait_for_completion(completion, 30000));

    // transform TSP's output data into layout expected by user
    for (size_t i = 0; i < numOutputs; ++i) {
        const auto &layout = outputTensorLayouts().at(i);

        uint8_t *input = nullptr;
        uint8_t *output = outputBuffers.at(i);
        size_t inputSize = layout.getIoSize();
        size_t outputSize = outputSizes.at(i);

        GROQOK(groq_get_data_handle(outputIoba, 0, &input));

        assert(input);
        assert(output);

        layout.toHost(input, inputSize, output, outputSize);
    }
}

const std::vector<TensorLayout> &SimpleRunner::inputTensorLayouts() const
{
    const auto &program = iop.getPrograms().at(programIndex);
    const auto &ep = program.getEntrypoints().at(entrypointIndex);
    const auto &input = ep.getInputIODescriptor();
    const auto &layouts = input.getTensorLayouts();
    return layouts;
}

const std::vector<TensorLayout> &SimpleRunner::outputTensorLayouts() const
{
    const auto &program = iop.getPrograms().at(programIndex);
    const auto &ep = program.getEntrypoints().at(entrypointIndex);
    const auto &input = ep.getOutputIODescriptor();
    const auto &layouts = input.getTensorLayouts();
    return layouts;
}

} // namespace groq
