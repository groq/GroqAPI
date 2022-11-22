#include "Misc.hpp"

#include <fstream>

std::vector<uint8_t> readFile(const std::string &filename)
{
    std::ifstream ifstream;
    ifstream.exceptions(std::ios_base::badbit | std::ios_base::failbit);
    ifstream.open(filename, std::ios_base::binary | std::ios_base::ate);

    const auto size = ifstream.tellg();
    ifstream.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size, 0x00);
    ifstream.read(reinterpret_cast<char *>(&data[0]), size);
    return data;
}

void writeFile(const std::string &filename, void *data, size_t n)
{
    std::ofstream out;
    out.open(filename);
    out.write(reinterpret_cast<const char *>(data), n);
};