#include "GpuConvolution.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void CheckCuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__global__ void ConvReluKernel(const unsigned char *image, const float *filter, double *outPlus, double *outMinus,
                               int imageWidth, int imageHeight, int channels, int filterHeight, int filterWidth,
                               int stride, int outWidth, int outHeight)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outWidth || oy >= outHeight) {
        return;
    }

    const int ix = ox * stride;
    const int iy = oy * stride;
    float sum = 0.0f;

    for (int fy = 0; fy < filterHeight; ++fy) {
        for (int fx = 0; fx < filterWidth; ++fx) {
            for (int c = 0; c < channels; ++c) {
                const int imageIndex = ((iy + fy) * imageWidth + (ix + fx)) * channels + c;
                const int filterIndex = (fy * filterWidth + fx) * channels + c;
                sum += static_cast<float>(image[imageIndex]) * filter[filterIndex];
            }
        }
    }

    const int outIndex = oy * outWidth + ox;
    outPlus[outIndex] = sum > 0.0f ? static_cast<double>(sum) : 0.0;
    outMinus[outIndex] = sum < 0.0f ? static_cast<double>(-sum) : 0.0;
}

std::vector<float> FlattenFilterFloat(const cv::Mat &filter)
{
    cv::Mat filter32f;
    if (filter.depth() != CV_32F) {
        filter.convertTo(filter32f, CV_MAKE_TYPE(CV_32F, filter.channels()));
    } else {
        filter32f = filter;
    }

    std::vector<float> values(static_cast<std::size_t>(filter32f.rows * filter32f.cols * filter32f.channels()));
    std::size_t offset = 0;
    for (int r = 0; r < filter32f.rows; ++r) {
        const float *row = filter32f.ptr<float>(r);
        for (int c = 0; c < filter32f.cols; ++c) {
            for (int ch = 0; ch < filter32f.channels(); ++ch) {
                values[offset++] = row[c * filter32f.channels() + ch];
            }
        }
    }
    return values;
}

PolarMaps LaunchPolarMaps(const unsigned char *dImage, int imageWidth, int imageHeight, int channels,
                          const cv::Mat &filter, int stride, const MapSize &outSize)
{
    const std::vector<float> hostFilter = FlattenFilterFloat(filter);
    float *dFilter = nullptr;
    double *dPlus = nullptr;
    double *dMinus = nullptr;

    const std::size_t outCount = static_cast<std::size_t>(outSize.width) * static_cast<std::size_t>(outSize.height);
    CheckCuda(cudaMalloc(&dFilter, hostFilter.size() * sizeof(float)), "cudaMalloc filter");
    CheckCuda(cudaMalloc(&dPlus, outCount * sizeof(double)), "cudaMalloc plus");
    CheckCuda(cudaMalloc(&dMinus, outCount * sizeof(double)), "cudaMalloc minus");
    CheckCuda(cudaMemcpy(dFilter, hostFilter.data(), hostFilter.size() * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy filter");

    const dim3 block(16, 16);
    const dim3 grid((outSize.width + block.x - 1) / block.x, (outSize.height + block.y - 1) / block.y);
    ConvReluKernel<<<grid, block>>>(dImage, dFilter, dPlus, dMinus, imageWidth, imageHeight, channels, filter.rows,
                                    filter.cols, stride, outSize.width, outSize.height);
    CheckCuda(cudaGetLastError(), "ConvReluKernel launch");
    CheckCuda(cudaDeviceSynchronize(), "ConvReluKernel sync");

    PolarMaps maps;
    maps.plus = cv::Mat(outSize.height, outSize.width, CV_64F);
    maps.minus = cv::Mat(outSize.height, outSize.width, CV_64F);
    CheckCuda(cudaMemcpy(maps.plus.ptr<double>(), dPlus, outCount * sizeof(double), cudaMemcpyDeviceToHost),
              "cudaMemcpy plus");
    CheckCuda(cudaMemcpy(maps.minus.ptr<double>(), dMinus, outCount * sizeof(double), cudaMemcpyDeviceToHost),
              "cudaMemcpy minus");

    cudaFree(dFilter);
    cudaFree(dPlus);
    cudaFree(dMinus);
    return maps;
}

} // namespace

bool IsConvolutionCudaBuilt()
{
    return true;
}

bool IsCudaDeviceAvailable(int device)
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return false;
    }
    return device >= 0 && device < count;
}

void EnsureCudaDevice(int device)
{
    if (!IsCudaDeviceAvailable(device)) {
        throw std::runtime_error("requested CUDA device is not available: " + std::to_string(device));
    }
    CheckCuda(cudaSetDevice(device), "cudaSetDevice");
}

BankMapsForImage ConvolveImageGpu(const cv::Mat &image, const std::vector<LoadedFilterBank> &banks,
                                  const std::vector<MapSize> &mapSizesPerBankFilter, int device)
{
    EnsureCudaDevice(device);

    if (image.empty() || image.depth() != CV_8U || !image.isContinuous()) {
        throw std::runtime_error("GPU convolution requires continuous CV_8U image");
    }

    const int channels = image.channels();
    const std::size_t imageBytes =
        static_cast<std::size_t>(image.rows) * static_cast<std::size_t>(image.cols) * static_cast<std::size_t>(channels);

    unsigned char *dImage = nullptr;
    CheckCuda(cudaMalloc(&dImage, imageBytes), "cudaMalloc image");
    CheckCuda(cudaMemcpy(dImage, image.ptr<unsigned char>(), imageBytes, cudaMemcpyHostToDevice),
              "cudaMemcpy image");

    BankMapsForImage imageMaps;
    imageMaps.banks.resize(banks.size());

    size_t globalFilterIndex = 0;
    try {
        for (size_t bankIndex = 0; bankIndex < banks.size(); ++bankIndex) {
            const LoadedFilterBank &bank = banks[bankIndex];
            FilterMapsForImage &bankMaps = imageMaps.banks[bankIndex];
            bankMaps.filters.resize(bank.filters.size());

            for (size_t filterIndex = 0; filterIndex < bank.filters.size(); ++filterIndex) {
                const MapSize &outSize = mapSizesPerBankFilter[globalFilterIndex];
                bankMaps.filters[filterIndex] =
                    LaunchPolarMaps(dImage, image.cols, image.rows, channels, bank.filters[filterIndex], bank.stride,
                                    outSize);
                ++globalFilterIndex;
            }
        }
    } catch (...) {
        cudaFree(dImage);
        throw;
    }

    cudaFree(dImage);
    return imageMaps;
}
