cmake_minimum_required(VERSION 3.16)

project(objects_ArNI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(../../../E/ArNI)

add_library(objects_ArNI SHARED

  objects_arni.cpp
  objects_arni.h
  RasterSpikeSource.cpp
)

target_compile_definitions(objects_ArNI PRIVATE OBJECTS_ARNI_LIBRARY)
