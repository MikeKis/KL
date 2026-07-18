project(convolution LANGUAGES CXX)

set(SRC
    "convolution.cpp"
    "ConvolutionConfig.cpp"
    "ConvolutionPipeline.cpp"
    "ConvolutionOutputWriter.cpp"
    "ConvolutionRunner.cpp"
    "FilterBank.cpp"
    "RawImageLoader.cpp"
    "SpikeProjector.cpp"
    "VmaxEstimator.cpp"
)

add_executable(${PROJECT_NAME} ${SRC})

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../Workplace"
)

target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tomlplusplus/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    target_sources(${PROJECT_NAME} PRIVATE "GpuConvolution.cu")
    target_compile_definitions(${PROJECT_NAME} PRIVATE CONVOLUTION_HAS_CUDA=1)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    message(STATUS "convolution: CUDA enabled")
else()
    target_sources(${PROJECT_NAME} PRIVATE "GpuConvolutionStub.cpp")
    message(STATUS "convolution: CUDA not found, CPU-only build")
endif()

if(NOT MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE -fpermissive)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${OpenCV_LIBS}
    PCFilters
)

option(CONVOLUTION_BUILD_TESTS "Build convolution unit tests" ON)
if(CONVOLUTION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
