cmake_minimum_required(VERSION 3.16) # Ubuntu 18.04 ships with 3.16.3

project(groq_c_runtime_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(/opt/groq/runtime/lib)
link_directories(/opt/groq/runtime/lib/groqruntime/lib)

add_executable(matmul_example
    matmul_example.cpp
    Device.cpp
    Driver.cpp
    IOP.cpp
    Misc.cpp
    SimpleRunner.cpp
)
target_link_libraries(matmul_example groqruntime)
