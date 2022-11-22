#pragma once

#include <groqio.h>
#include <string>
#include <string_view>
#include <vector>

namespace groq {

class TensorLayout
{
    ::TensorLayout layout{ nullptr };
    std::string name{};
    size_t iodSize; // tsp size, not for just this layout -- but for this one and its "neighbors"
    size_t size;    // host size
    int32_t format;
    std::vector<uint32_t> dimensions;

public:
    TensorLayout(::TensorLayout layout, std::string_view name, size_t iodSize);
    TensorLayout() = default;

    const std::string &getName() const { return name; }
    enum Format { STRIDED = 0, CONTIGUOUS = 1 };
    Format getFormat() const { return static_cast<Format>(format); }
    size_t getHostSize() const { return size; }
    size_t getIoSize() const { return iodSize; }
    const std::vector<uint32_t> &getDimensions() const { return dimensions; }

    void toHost(uint8_t *input, size_t inputSize, uint8_t *output, size_t outputSize) const;
    void fromHost(uint8_t *input, size_t inputSize, uint8_t *output, size_t outputSize) const;
};

class IODescriptor
{
    std::vector<TensorLayout> layouts;
    size_t size;

public:
    IODescriptor() = default;
    IODescriptor(::IODescriptor iodescriptor, size_t size);
    const std::vector<TensorLayout> &getTensorLayouts() const { return layouts; }
    size_t getSize() const { return size; }
};

class EntryPoint
{
    std::string name;
    IODescriptor input;
    IODescriptor output;

public:
    EntryPoint(::EntryPoint entrypoint, std::string_view name);

    const std::string &getName() const { return name; }
    const IODescriptor &getInputIODescriptor() const { return input; }
    const IODescriptor &getOutputIODescriptor() const { return output; }
};

class Program
{
    std::vector<EntryPoint> entrypoints;
    std::string name;

public:
    Program(const ::Program program, std::string_view name);
    const std::string &getName() const { return name; }
    const std::vector<EntryPoint> &getEntrypoints() const { return entrypoints; }
    size_t numEntrypoints() const { return entrypoints.size(); }
};

class IOP
{
    ::IOP iop;
    std::vector<uint8_t> data;
    std::vector<Program> programs;

public:
    IOP(const std::string &filename);
    IOP(const uint8_t *buffer, size_t size);
    ~IOP();

    ::IOP handle() const { return iop; }
    const std::vector<Program> &getPrograms() const { return programs; }
    size_t numPrograms() const { return programs.size(); }

private:
    void initialize();
};

} // namespace groq
