project(unsfilc CXX)

################################################################################
# Source groups
################################################################################
set(SRC
    "unsfilc.cpp"
)

source_group("Source Files" FILES ${SRC})

set(HDR
    "../UnsupervisedFiltersMultiple/UnsupervisedFiltersMultiple.h"
)

source_group("Header Files" FILES ${HDR})

set(ALL_FILES
    ${SRC}
    ${HDR}
)

################################################################################
# Target
################################################################################
add_executable(${PROJECT_NAME} ${ALL_FILES}
    )

################################################################################
# Output directory
################################################################################
set_target_properties(${PROJECT_NAME} PROPERTIES
RUNTIME_OUTPUT_DIRECTORY    "${CMAKE_CURRENT_SOURCE_DIR}/../Workplace"
)

################################################################################
# Compile definitions
################################################################################

target_compile_options(${PROJECT_NAME} PRIVATE
            -fpermissive
            )

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${OpenCV_LIBS}
    UnsupervisedFiltersMultiple
)
