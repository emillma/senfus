cmake_minimum_required(VERSION 3.22)
project(example LANGUAGES CXX)

include_directories(include)

set(Python_ROOT_DIR "/root/miniconda3/envs/dev")
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "Python version: ${Python_VERSION}")


include_directories(include/eigen)
add_subdirectory(include/pybind11)

add_library(mylib SHARED generated/bindings.cpp)
target_include_directories(mylib PUBLIC include/pybind11/include)
target_link_libraries(mylib PUBLIC Python::Python)
set_target_properties(mylib PROPERTIES PREFIX "")
