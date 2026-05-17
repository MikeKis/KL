#include "ConvolutionConfig.h"
#include "ConvolutionOutputWriter.h"
#include "ConvolutionPipeline.h"
#include "FilterBank.h"
#include "RawImageLoader.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

const char kUsage[] = "Usage: convolution <config.toml>\n";

int RunConvolution(const char *configPath)
{
    const ConvolutionConfig config = ParseConvolutionConfig(configPath);

    std::vector<std::string> filterFiles;
    std::vector<int> strides;
    filterFiles.reserve(config.filters.size());
    strides.reserve(config.filters.size());
    for (const FilterConfigEntry &entry : config.filters) {
        filterFiles.push_back(entry.file);
        strides.push_back(entry.stride);
    }

    const std::vector<LoadedFilterBank> banks = LoadFilterBanks(filterFiles, strides);
    const int channels = GetFilterChannels(banks);
    const std::vector<cv::Mat> images =
        LoadRawImages(config.inputFile, config.inputWidth, config.inputHeight, channels);

    std::cerr << "Loaded " << images.size() << " images, channels=" << channels << '\n';

    ConvolutionPipelineResult pipeline = BuildPipelineResult(images, banks);
    std::vector<VmaxEstimateResult> vmaxResults =
        EstimateAllVmax(pipeline, config.projectionMode, config.projectionCriterion);

    WriteConvolutionOutputs(config, pipeline, vmaxResults, configPath, static_cast<int>(images.size()));
    return 0;
}

} // namespace

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << kUsage;
        return 1;
    }

    try {
        return RunConvolution(argv[1]);
    } catch (const std::exception &ex) {
        std::cerr << "convolution error: " << ex.what() << '\n';
        return 1;
    }
}
