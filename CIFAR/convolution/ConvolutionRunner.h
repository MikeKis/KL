#pragma once

#include "FilterBank.h"

#include <opencv2/core.hpp>

#include <utility>
#include <vector>

struct MapSize {
    int width = 0;
    int height = 0;
};

MapSize ComputeOutputMapSize(int imageWidth, int imageHeight, int filterWidth, int filterHeight, int stride);

struct PolarMaps {
    cv::Mat plus;
    cv::Mat minus;
};

struct FilterMapsForImage {
    std::vector<PolarMaps> filters;
};

struct BankMapsForImage {
    std::vector<FilterMapsForImage> banks;
};

struct ConvolutionRunResult {
    std::vector<BankMapsForImage> images;
    std::vector<MapSize> mapSizesPerBankFilter;
};

ConvolutionRunResult RunConvolutions(const std::vector<cv::Mat> &images,
                                     const std::vector<LoadedFilterBank> &banks);
