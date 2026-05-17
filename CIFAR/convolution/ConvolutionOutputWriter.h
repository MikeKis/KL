#pragma once

#include "ConvolutionConfig.h"
#include "ConvolutionPipeline.h"

void WriteConvolutionOutputs(const ConvolutionConfig &config,
                             const ConvolutionPipelineResult &pipeline,
                             const std::vector<VmaxEstimateResult> &vmaxResults,
                             const std::string &configPath,
                             int imageCount);
