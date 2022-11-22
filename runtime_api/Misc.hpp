#pragma once

#include <groqio.h>

#include <stdexcept>
#include <string>
#include <vector>

#define unused(x) (void)x;
#define STRINGIFY(x) #x
#define GROQOK(x)                                                                                                      \
    {                                                                                                                  \
        Status status = (x);                                                                                           \
        if (status != GROQ_SUCCESS) {                                                                                  \
            throw std::runtime_error("Error " + std::to_string(status) + ": " + STRINGIFY(x));                         \
        }                                                                                                              \
    }

std::vector<uint8_t> readFile(const std::string &filename);
void writeFile(const std::string &filename, void *data, size_t n);
