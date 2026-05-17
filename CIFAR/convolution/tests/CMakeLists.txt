project(convolution_test CXX)

set(SRC
    "convolution_test.cpp"
    "../ConvolutionConfig.cpp"
    "../ConvolutionRunner.cpp"
    "../SpikeProjector.cpp"
    "../VmaxEstimator.cpp"
)

add_executable(${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tomlplusplus/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -fpermissive
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${OpenCV_LIBS}
)

add_test(NAME convolution_test COMMAND convolution_test
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
