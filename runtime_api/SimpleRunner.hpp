#pragma once

#include <groqio.h>

#include <vector>

namespace groq {

class Device;
class Driver;
class IOP;
class TensorLayout;

class SimpleRunner
{
    const IOP &iop;
    const size_t programIndex;
    const size_t entrypointIndex;
    const size_t tspInputSize;
    const size_t tspOutputSize;
    const size_t numInputs;
    const size_t numOutputs;

    ::IOBufferArray inputIoba;
    ::IOBufferArray outputIoba;
    std::vector<uint8_t *> inputBuffers;
    std::vector<uint8_t *> outputBuffers;
    std::vector<size_t> inputSizes;
    std::vector<size_t> outputSizes;

public:
    SimpleRunner(Driver &driver, const IOP &iop, size_t programIndex = 0, size_t entrypointIndex = 0);
    ~SimpleRunner();

    void addInputBuffer(uint8_t *buffer, size_t size, size_t index);
    void addOutputBuffer(uint8_t *buffer, size_t size, size_t index);
    void invoke(Device &device);

private:
    const std::vector<TensorLayout> &inputTensorLayouts() const;
    const std::vector<TensorLayout> &outputTensorLayouts() const;
};

} // namespace groq
