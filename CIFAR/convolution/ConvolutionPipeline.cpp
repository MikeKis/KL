#include "ConvolutionPipeline.h"

#include "ConvolutionOutputWriter.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

static void AppendMapValues(std::vector<double> &target, const cv::Mat &map)
{
    for (int y = 0; y < map.rows; ++y) {
        for (int x = 0; x < map.cols; ++x) {
            target.push_back(map.at<double>(y, x));
        }
    }
}

ConvolutionPipelineMetadata BuildPipelineMetadata(const std::vector<LoadedFilterBank> &banks, int imageWidth,
                                                int imageHeight)
{
    ConvolutionPipelineMetadata metadata;
    metadata.mapSizesPerBankFilter = ComputeMapSizesForBanks(imageWidth, imageHeight, banks);

    size_t globalFilterIndex = 0;
    for (size_t bankIndex = 0; bankIndex < banks.size(); ++bankIndex) {
        const LoadedFilterBank &bank = banks[bankIndex];
        for (size_t filterIndex = 0; filterIndex < bank.filters.size(); ++filterIndex) {
            PhysicalFilterDescriptor descriptor;
            descriptor.sourceFile = bank.file;
            descriptor.filterIndex = static_cast<int>(filterIndex);
            descriptor.mapSize = metadata.mapSizesPerBankFilter[globalFilterIndex];
            metadata.filters.push_back(descriptor);
            ++globalFilterIndex;
        }
    }
    return metadata;
}

std::vector<std::size_t> SelectVmaxSampleIndices(std::size_t imageCount, int vmaxSampleSize)
{
    if (imageCount == 0) {
        throw std::runtime_error("cannot select Vmax sample from empty image set");
    }
    if (vmaxSampleSize <= 0) {
        throw std::invalid_argument("vmax_sample_size must be positive");
    }

    std::vector<std::size_t> indices(imageCount);
    for (std::size_t i = 0; i < imageCount; ++i) {
        indices[i] = i;
    }

    const std::size_t sampleCount = std::min(imageCount, static_cast<std::size_t>(vmaxSampleSize));
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(sampleCount);
    std::sort(indices.begin(), indices.end());
    return indices;
}

void AccumulateVmaxValues(const BankMapsForImage &imageMaps, std::vector<std::vector<double>> &valuesPerFilter)
{
    size_t globalFilterIndex = 0;
    for (const FilterMapsForImage &bankMaps : imageMaps.banks) {
        for (const PolarMaps &maps : bankMaps.filters) {
            AppendMapValues(valuesPerFilter[globalFilterIndex], maps.plus);
            AppendMapValues(valuesPerFilter[globalFilterIndex], maps.minus);
            ++globalFilterIndex;
        }
    }
}

ConvolutionPipelineResult BuildPipelineResult(const std::vector<cv::Mat> &images,
                                            const std::vector<LoadedFilterBank> &banks)
{
    ConvolutionPipelineResult pipeline;
    pipeline.convolutions = RunConvolutions(images, banks);

    const ConvolutionPipelineMetadata metadata =
        BuildPipelineMetadata(banks, images.front().cols, images.front().rows);
    pipeline.filters = metadata.filters;
    pipeline.valuesPerFilter.resize(pipeline.filters.size());

    for (const BankMapsForImage &imageMaps : pipeline.convolutions.images) {
        AccumulateVmaxValues(imageMaps, pipeline.valuesPerFilter);
    }
    return pipeline;
}

std::vector<VmaxEstimateResult> EstimateAllVmax(const std::vector<std::vector<double>> &valuesPerFilter,
                                                ProjectionMode mode,
                                                double criterion)
{
    std::vector<VmaxEstimateResult> results;
    results.reserve(valuesPerFilter.size());
    int i = 0;
    for (const std::vector<double> &values : valuesPerFilter) {
        results.push_back(EstimateVmax(values, mode, criterion));
        std::cout << ++i << " out of " << valuesPerFilter.size() << " processed\n";
    }
    return results;
}

void RunConvolutionPipeline(const ConvolutionConfig &config, const char *configPath,
                            const std::vector<cv::Mat> &images, const std::vector<LoadedFilterBank> &banks)
{
    const ConvolutionPipelineMetadata metadata =
        BuildPipelineMetadata(banks, config.inputWidth, config.inputHeight);
    const std::vector<std::size_t> vmaxSampleIndices =
        SelectVmaxSampleIndices(images.size(), config.vmaxSampleSize);

    std::cout << "Images: " << images.size() << ", Vmax sample: " << vmaxSampleIndices.size() << '\n';

    std::vector<std::vector<double>> valuesPerFilter(metadata.filters.size());
    for (std::size_t sampleOrdinal = 0; sampleOrdinal < vmaxSampleIndices.size(); ++sampleOrdinal) {
        const std::size_t imageIndex = vmaxSampleIndices[sampleOrdinal];
        std::cout << "Vmax sampling: image " << (sampleOrdinal + 1) << '/' << vmaxSampleIndices.size()
                  << " (index " << imageIndex << ")\n";

        const BankMapsForImage imageMaps =
            ConvolveImage(images[imageIndex], banks, metadata.mapSizesPerBankFilter);
        AccumulateVmaxValues(imageMaps, valuesPerFilter);
    }

    std::cout << "Estimating Vmax for " << valuesPerFilter.size() << " filters\n";
    const std::vector<VmaxEstimateResult> vmaxResults =
        EstimateAllVmax(valuesPerFilter, config.projectionMode, config.projectionCriterion);

    ConvolutionOutputSession session(config, metadata.filters, vmaxResults, configPath,
                                     static_cast<int>(images.size()), static_cast<int>(vmaxSampleIndices.size()));

    for (std::size_t imageIndex = 0; imageIndex < images.size(); ++imageIndex) {
        std::cout << "Writing output: image " << (imageIndex + 1) << '/' << images.size() << '\n';
        const BankMapsForImage imageMaps =
            ConvolveImage(images[imageIndex], banks, metadata.mapSizesPerBankFilter);
        session.WriteImage(imageMaps);
    }

    session.Finish();
    std::cout << "Done.\n";
}
