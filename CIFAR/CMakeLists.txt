cmake_minimum_required(VERSION 3.14)
project(CIFAR CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_subdirectory(PCFilters)
add_subdirectory(pcfilCIFAR)
add_subdirectory(convolution)
