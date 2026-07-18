#pragma once

#include "ConvolutionRunner.h"
#include "FilterBank.h"

#include <opencv2/core.hpp>

#include <vector>

bool IsConvolutionCudaBuilt();
bool IsCudaDeviceAvailable(int device);
void EnsureCudaDevice(int device);

BankMapsForImage ConvolveImageGpu(const cv::Mat &image, const std::vector<LoadedFilterBank> &banks,
                                  const std::vector<MapSize> &mapSizesPerBankFilter, int device);
