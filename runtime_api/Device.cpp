#include "Device.hpp"

#include "Misc.hpp"
#include "IOP.hpp"

namespace groq {

Device::Device(const ::Device device)
    : device(device)
    , numaNode(-1)
{
}

Device::~Device()
{
    auto status = ::groq_device_close(device);
    unused(status);
}

::Device Device::handle() const
{
    return device;
}

void Device::open()
{
    GROQOK(::groq_device_open(device));
    GROQOK(::groq_device_numa_node(device, &numaNode));
}

bool Device::isOpen()
{
    bool open = false;
    GROQOK(::groq_device_is_open(device, &open));
    return open;
}

void Device::close()
{
    GROQOK(::groq_device_close(device));
}

void Device::clearMemory()
{
    GROQOK(::groq_device_clear_memory(device));
}

void Device::reset()
{
    GROQOK(groq_device_reset(device));
}

int32_t Device::getNumaNode() const
{
    return numaNode;
}

void Device::loadProgram(const IOP &iop, size_t n, bool keepEntryPoints)
{
    GROQOK(groq_load_program(device, iop.handle(), n, keepEntryPoints));
}

} // namespace groq