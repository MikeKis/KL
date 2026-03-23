cmake_minimum_required(VERSION 3.14)
project(UnsupervisedFiltersMultiple CXX)

################################################################################
# Source groups
################################################################################
source_group("" FILES)

set(Header_Files
    "UnsupervisedFiltersMultiple.h"
)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files
    "UnsupervisedFiltersMultiple.cpp"
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${no_group_source_files}
    ${Header_Files}
    ${Source_Files}
)

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} STATIC ${ALL_FILES})

set(ROOT_NAMESPACE UnsupervisedFiltersMultiple)

set_target_properties(${PROJECT_NAME} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY    "${CMAKE_CURRENT_SOURCE_DIR}/../lib"
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION            "TRUE"
)

################################################################################
# Compile definitions
################################################################################
target_compile_definitions(${PROJECT_NAME} PRIVATE
    "FOR_LINUX"
    "_LIB"
)

target_compile_options(${PROJECT_NAME} PRIVATE
            -fpermissive
            )

