#pragma once

#include <groqio.h>

namespace groq {

class IOP;
class Program;

/**
 * @brief groq::Device is a wrapper around the ::Device pointer from groqio.h
 *
 * This wrapper is intended to relieve programmers of the burden of writing
 * error-checking code around each call into the groqio API.  If any of the
 * operations fail, an exception will be thrown.
 */
class Device
{
    ::Device device;
    int32_t numaNode{ -1 };

public:
    Device(const ::Device device);
    ~Device();

    ::Device handle() const;

    void open();
    bool isOpen();
    void close();

    void clearMemory();
    void reset();

    int32_t getNumaNode() const;

    void loadProgram(const IOP &iop, size_t n, bool keepEntryPoints);
};

} // namespace groq
