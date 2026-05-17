#pragma once

#include "ConvolutionConfig.h"
#include "ConvolutionRunner.h"
#include "FilterBank.h"
#include "VmaxEstimator.h"

#include <string>
#include <vector>

struct PhysicalFilterDescriptor {
    std::string sourceFile;
    int filterIndex = 0;
    MapSize mapSize;
};

struct ConvolutionPipelineResult {
    ConvolutionRunResult convolutions;
    std::vector<PhysicalFilterDescriptor> filters;
    std::vector<std::vector<double>> valuesPerFilter;
    std::vector<VmaxEstimateResult> vmaxResults;
};

ConvolutionPipelineResult BuildPipelineResult(const std::vector<cv::Mat> &images,
                                            const std::vector<LoadedFilterBank> &banks);

std::vector<VmaxEstimateResult> EstimateAllVmax(const ConvolutionPipelineResult &pipeline,
                                                ProjectionMode mode,
                                                double criterion);
