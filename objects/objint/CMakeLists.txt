project(objint CXX)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

################################################################################
# Source groups
################################################################################
set(SRC
    "fromRaster.cpp"
    "objint.cpp"
    "objpre.cpp"
    "objlea.cpp"
)

source_group("Source Files" FILES ${SRC})

set(HDR
    "objint.h"
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
    objfix.cpp
    objinf.cpp
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
    zmq
)
