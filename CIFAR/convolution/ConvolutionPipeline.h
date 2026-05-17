#pragma once

#include "ConvolutionConfig.h"
#include "ConvolutionRunner.h"
#include "FilterBank.h"
#include "VmaxEstimator.h"

#include <cstddef>
#include <string>
#include <vector>

struct PhysicalFilterDescriptor {
    std::string sourceFile;
    int filterIndex = 0;
    MapSize mapSize;
};

struct ConvolutionPipelineMetadata {
    std::vector<PhysicalFilterDescriptor> filters;
    std::vector<MapSize> mapSizesPerBankFilter;
};

struct ConvolutionPipelineResult {
    ConvolutionRunResult convolutions;
    std::vector<PhysicalFilterDescriptor> filters;
    std::vector<std::vector<double>> valuesPerFilter;
};

ConvolutionPipelineMetadata BuildPipelineMetadata(const std::vector<LoadedFilterBank> &banks, int imageWidth,
                                                int imageHeight);

std::vector<std::size_t> SelectVmaxSampleIndices(std::size_t imageCount, int vmaxSampleSize);

void AccumulateVmaxValues(const BankMapsForImage &imageMaps, std::vector<std::vector<double>> &valuesPerFilter);

ConvolutionPipelineResult BuildPipelineResult(const std::vector<cv::Mat> &images,
                                            const std::vector<LoadedFilterBank> &banks);

std::vector<VmaxEstimateResult> EstimateAllVmax(const std::vector<std::vector<double>> &valuesPerFilter,
                                                ProjectionMode mode,
                                                double criterion);

void RunConvolutionPipeline(const ConvolutionConfig &config, const char *configPath,
                            const std::vector<cv::Mat> &images, const std::vector<LoadedFilterBank> &banks);
