cmake_minimum_required(VERSION 3.16)

project(objects_ArNI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(../../../E/ArNI)
link_directories(../../../E/ArNI/lib)

add_library(objects_ArNI SHARED

  objects_arni.cpp
  objects_arni.h
  RasterSpikeSource.cpp
  objects_arni_int.h
  GetSpikesfromImage.cpp
  LabelSpikeSource.cpp
  ../pugixml-1.10/src/pugixml.cpp
)

target_compile_definitions(objects_ArNI PRIVATE OBJECTS_ARNI_LIBRARY)

set_target_properties(${PROJECT_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY    "${CMAKE_CURRENT_SOURCE_DIR}/../Experiments"
        PREFIX ""
)
