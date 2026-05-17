#pragma once

#include "ConvolutionConfig.h"
#include "ConvolutionPipeline.h"
#include "ConvolutionRunner.h"
#include "VmaxEstimator.h"

#include <fstream>
#include <string>
#include <vector>

class ConvolutionOutputSession {
public:
    ConvolutionOutputSession(const ConvolutionConfig &config,
                             const std::vector<PhysicalFilterDescriptor> &filters,
                             const std::vector<VmaxEstimateResult> &vmaxResults,
                             const std::string &configPath,
                             int imageCount,
                             int vmaxSampleCount);

    void WriteImage(const BankMapsForImage &imageMaps);
    void Finish();

private:
    const ConvolutionConfig &config_;
    const std::vector<PhysicalFilterDescriptor> &filters_;
    const std::vector<VmaxEstimateResult> &vmaxResults_;
    std::ofstream binStream_;
    std::ofstream csvStream_;
    std::ofstream logStream_;
};

void WriteConvolutionOutputs(const ConvolutionConfig &config,
                             const ConvolutionPipelineResult &pipeline,
                             const std::vector<VmaxEstimateResult> &vmaxResults,
                             const std::string &configPath,
                             int imageCount);
