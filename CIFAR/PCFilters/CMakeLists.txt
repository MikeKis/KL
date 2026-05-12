cmake_minimum_required(VERSION 3.14)
project(PCFilters CXX)

set(Header_Files
    "PCFilters.h"
)

set(Source_Files
    "PCFilters.cpp"
)

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
)

add_library(${PROJECT_NAME} STATIC ${ALL_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../lib"
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    "FOR_LINUX"
    "_LIB"
)

target_include_directories(${PROJECT_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${OpenCV_LIBS}
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -fpermissive
)
