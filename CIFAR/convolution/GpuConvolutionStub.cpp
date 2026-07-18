#include "GpuConvolution.h"

#include <stdexcept>
#include <string>

#ifndef CONVOLUTION_HAS_CUDA

bool IsConvolutionCudaBuilt()
{
    return false;
}

bool IsCudaDeviceAvailable(int /*device*/)
{
    return false;
}

void EnsureCudaDevice(int /*device*/)
{
    throw std::runtime_error("use_gpu=true but convolution was built without CUDA support");
}

BankMapsForImage ConvolveImageGpu(const cv::Mat & /*image*/, const std::vector<LoadedFilterBank> & /*banks*/,
                                  const std::vector<MapSize> & /*mapSizesPerBankFilter*/, int /*device*/)
{
    throw std::runtime_error("use_gpu=true but convolution was built without CUDA support");
}

#endif
