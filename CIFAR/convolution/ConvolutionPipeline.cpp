#include "ConvolutionPipeline.h"

#include <stdexcept>

static void AppendMapValues(std::vector<double> &target, const cv::Mat &map)
{
    for (int y = 0; y < map.rows; ++y) {
        for (int x = 0; x < map.cols; ++x) {
            target.push_back(map.at<double>(y, x));
        }
    }
}

ConvolutionPipelineResult BuildPipelineResult(const std::vector<cv::Mat> &images,
                                            const std::vector<LoadedFilterBank> &banks)
{
    ConvolutionPipelineResult pipeline;
    pipeline.convolutions = RunConvolutions(images, banks);
    pipeline.valuesPerFilter.resize(pipeline.convolutions.mapSizesPerBankFilter.size());

    size_t globalFilterIndex = 0;
    for (size_t bankIndex = 0; bankIndex < banks.size(); ++bankIndex) {
        const LoadedFilterBank &bank = banks[bankIndex];
        for (size_t filterIndex = 0; filterIndex < bank.filters.size(); ++filterIndex) {
            PhysicalFilterDescriptor descriptor;
            descriptor.sourceFile = bank.file;
            descriptor.filterIndex = static_cast<int>(filterIndex);
            descriptor.mapSize = pipeline.convolutions.mapSizesPerBankFilter[globalFilterIndex];
            pipeline.filters.push_back(descriptor);

            std::vector<double> &values = pipeline.valuesPerFilter[globalFilterIndex];
            for (const BankMapsForImage &imageMaps : pipeline.convolutions.images) {
                const PolarMaps &maps = imageMaps.banks[bankIndex].filters[filterIndex];
                AppendMapValues(values, maps.plus);
                AppendMapValues(values, maps.minus);
            }
            ++globalFilterIndex;
        }
    }

    return pipeline;
}

std::vector<VmaxEstimateResult> EstimateAllVmax(const ConvolutionPipelineResult &pipeline,
                                                ProjectionMode mode,
                                                double criterion)
{
    std::vector<VmaxEstimateResult> results;
    results.reserve(pipeline.valuesPerFilter.size());

    for (const std::vector<double> &values : pipeline.valuesPerFilter) {
        results.push_back(EstimateVmax(values, mode, criterion));
    }
    return results;
}
