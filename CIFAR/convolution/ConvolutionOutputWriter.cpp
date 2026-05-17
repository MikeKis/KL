#include "ConvolutionOutputWriter.h"

#include "SpikeProjector.h"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace {

const char kAppVersion[] = "1.0.0";

void WriteMapValues(std::ostream &binStream, const cv::Mat &map)
{
    for (int y = 0; y < map.rows; ++y) {
        for (int x = 0; x < map.cols; ++x) {
            const double value = map.at<double>(y, x);
            binStream.write(reinterpret_cast<const char *>(&value), sizeof(double));
        }
    }
}

void WriteMapSpikes(std::ostream &csvStream, const cv::Mat &map, double vmax, bool &firstField)
{
    for (int y = 0; y < map.rows; ++y) {
        for (int x = 0; x < map.cols; ++x) {
            if (!firstField) {
                csvStream << ',';
            }
            firstField = false;
            csvStream << ProjectSpike(map.at<double>(y, x), vmax);
        }
    }
}

} // namespace

void WriteConvolutionOutputs(const ConvolutionConfig &config,
                             const ConvolutionPipelineResult &pipeline,
                             const std::vector<VmaxEstimateResult> &vmaxResults,
                             const std::string &configPath,
                             int imageCount)
{
    std::ofstream binStream(config.outputConvolutionsBin, std::ios::binary | std::ios::trunc);
    if (!binStream) {
        throw std::runtime_error("cannot open output file: " + config.outputConvolutionsBin);
    }

    std::ofstream csvStream(config.outputSpikesCsv, std::ios::trunc);
    if (!csvStream) {
        throw std::runtime_error("cannot open output file: " + config.outputSpikesCsv);
    }

    std::ofstream logStream(config.outputLog, std::ios::trunc);
    if (!logStream) {
        throw std::runtime_error("cannot open output file: " + config.outputLog);
    }

    logStream << "version=" << kAppVersion << '\n';
    logStream << "config=" << configPath << '\n';
    logStream << "images=" << imageCount << '\n';
    logStream << "mode=" << ProjectionModeToString(config.projectionMode) << '\n';
    logStream << "criterion=" << config.projectionCriterion << '\n';

    for (size_t filterIndex = 0; filterIndex < pipeline.filters.size(); ++filterIndex) {
        const PhysicalFilterDescriptor &descriptor = pipeline.filters[filterIndex];
        logStream << "map file=" << descriptor.sourceFile << " filter=" << descriptor.filterIndex
                  << " width=" << descriptor.mapSize.width << " height=" << descriptor.mapSize.height << '\n';
    }

    for (const BankMapsForImage &imageMaps : pipeline.convolutions.images) {
        size_t globalFilterIndex = 0;
        bool firstCsvField = true;
        for (size_t bankIndex = 0; bankIndex < imageMaps.banks.size(); ++bankIndex) {
            const FilterMapsForImage &bankMaps = imageMaps.banks[bankIndex];
            for (size_t filterIndex = 0; filterIndex < bankMaps.filters.size(); ++filterIndex) {
                const PolarMaps &maps = bankMaps.filters[filterIndex];
                const double vmax = vmaxResults[globalFilterIndex].vmax;

                WriteMapValues(binStream, maps.plus);
                WriteMapValues(binStream, maps.minus);
                WriteMapSpikes(csvStream, maps.plus, vmax, firstCsvField);
                WriteMapSpikes(csvStream, maps.minus, vmax, firstCsvField);
                ++globalFilterIndex;
            }
        }
        csvStream << '\n';
    }

    for (size_t filterIndex = 0; filterIndex < pipeline.filters.size(); ++filterIndex) {
        const PhysicalFilterDescriptor &descriptor = pipeline.filters[filterIndex];
        const VmaxEstimateResult &estimate = vmaxResults[filterIndex];
        logStream << "filter file=" << descriptor.sourceFile << " index=" << descriptor.filterIndex
                  << " Vmax=" << estimate.vmax;
        if (config.projectionMode == ProjectionMode::DefSparsity) {
            logStream << " saturation_fraction=" << estimate.saturationFraction;
        } else {
            logStream << " mean_spike=" << estimate.meanSpike;
        }
        logStream << '\n';
    }
}
