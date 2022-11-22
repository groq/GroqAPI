#pragma once

#include <groqio.h>

namespace groq {

class Device;
class Driver
{
    ::Driver driver;

public:
    Driver();
    ~Driver();

    ::Driver handle() const;
    Device getDevice(size_t n);
    Device getNextDevice();
};

} // namespace groq
