project(unsfil2app CXX)

set(SRC
    "unsfil2app.cpp"
)

add_executable(${PROJECT_NAME} ${SRC})

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../Workplace"
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -fpermissive
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${OpenCV_LIBS}
    PCFilters
    UnsupervisedFiltersMultiple
)
