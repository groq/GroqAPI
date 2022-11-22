#include "IOP.hpp"
#include "Misc.hpp"

#include <groqio.h>

#include <cstring>
#include <cassert>
#include <iostream>

namespace groq {

TensorLayout::TensorLayout(::TensorLayout layout, std::string_view name, size_t iodSize)
    : layout(layout)
    , name(name)
    , iodSize(iodSize)
{
    size_t nDims = 0;
    GROQOK(groq_tensor_layout_get_number_of_dimensions(layout, &nDims));
    GROQOK(groq_tensor_layout_get_size(layout, &size));
    GROQOK(groq_tensor_layout_get_format(layout, &format));
    for (size_t nth = 0; nth < nDims; ++nth) {
        uint32_t dimension;
        GROQOK(groq_tensor_layout_get_nth_dimension(layout, nth, &dimension));
        dimensions.push_back(dimension);
    }
}

void TensorLayout::toHost(uint8_t *input, size_t inputSize, uint8_t *output, size_t outputSize) const
{
    if (inputSize != getIoSize()) {
        throw std::runtime_error("Size mismatch");
    }
    if (outputSize != getHostSize()) {
        throw std::runtime_error("Size mismatch");
    }
    GROQOK(groq_tensor_layout_to_host(layout, input, inputSize, output, outputSize));
}

void TensorLayout::fromHost(uint8_t *input, size_t inputSize, uint8_t *output, size_t outputSize) const
{
    if (inputSize != getHostSize()) {
        throw std::runtime_error("Input size mismatch; expected " + std::to_string(getHostSize()) + " got "
                                 + std::to_string(inputSize));
    }
    if (outputSize != getIoSize()) {
        throw std::runtime_error("Output size mismatch; expected " + std::to_string(getIoSize()) + " got "
                                 + std::to_string(outputSize));
    }
    GROQOK(groq_tensor_layout_from_host(layout, input, inputSize, output, outputSize));
}

IODescriptor::IODescriptor(::IODescriptor iodescriptor, size_t size)
    : size(size)
{
    size_t n = 0;
    GROQOK(groq_iodescriptor_get_number_of_tensor_layouts(iodescriptor, &n));
    for (size_t nth = 0; nth < n; ++nth) {
        ::TensorLayout layout;
        char *name;
        GROQOK(groq_iodescriptor_get_nth_tensor_layout(iodescriptor, nth, &layout));
        GROQOK(groq_tensor_layout_get_name(layout, &name));
        layouts.emplace_back(layout, name, size);
    }
}

EntryPoint::EntryPoint(::EntryPoint entrypoint, std::string_view name)
    : name(name)
{
    ::IODescriptor inputIod, outputIod;
    GROQOK(groq_entrypoint_get_input_iodescriptor(entrypoint, &inputIod));
    GROQOK(groq_entrypoint_get_output_iodescriptor(entrypoint, &outputIod));

    size_t inputSize, outputSize;
    GROQOK(groq_entrypoint_get_input_size(entrypoint, &inputSize));
    GROQOK(groq_entrypoint_get_output_size(entrypoint, &outputSize));

    input = IODescriptor(inputIod, inputSize);
    output = IODescriptor(outputIod, outputSize);
}

Program::Program(const ::Program program, std::string_view name)
    : name(name)
{
    size_t n = 0;
    GROQOK(groq_get_number_of_entrypoints(program, &n));
    for (size_t nth = 0; nth < n; ++nth) {
        ::EntryPoint entrypoint;
        char *name;
        GROQOK(groq_get_nth_entrypoint(program, nth, &entrypoint));
        GROQOK(groq_entrypoint_get_name(entrypoint, &name));
        entrypoints.emplace_back(entrypoint, name);
    }
}

IOP::IOP(const std::string &filename)
    : data(readFile(filename))
{
    initialize();
}

IOP::IOP(const uint8_t *buffer, size_t size)
{
    data.resize(size);
    std::memcpy(&data[0], buffer, size);

    initialize();
}

IOP::~IOP()
{
    auto status = groq_iop_deinit(iop);
    unused(status);
}

void IOP::initialize()
{
    unsigned int n = 0; // TODO: size_t
    GROQOK(groq_iop_init(&data[0], data.size(), &iop));
    GROQOK(groq_iop_get_number_of_programs(iop, &n));
    for (size_t nth = 0; nth < n; ++nth) {
        ::Program program;
        char *name = nullptr;
        GROQOK(groq_get_nth_program(iop, nth, &program));
        GROQOK(groq_program_name(iop, nth, &name));
        programs.emplace_back(program, name);
    }
}

} // namespace groq
