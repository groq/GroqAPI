#include "Driver.hpp"

#include "Device.hpp"
#include "Misc.hpp"

namespace groq {

Driver::Driver()
{
    GROQOK(::groq_init(&driver));
}

Driver::~Driver()
{
    auto status = ::groq_deinit(&driver);
    unused(status);
}

::Driver Driver::handle() const
{
    return driver;
}

Device Driver::getDevice(size_t n)
{
    ::Device device;
    GROQOK(::groq_get_nth_device(driver, n, &device));
    return groq::Device(device);
}

Device Driver::getNextDevice()
{
    ::Device device;
    GROQOK(::groq_get_next_available_device(driver, &device));
    return groq::Device(device);
}

} // namespace groq