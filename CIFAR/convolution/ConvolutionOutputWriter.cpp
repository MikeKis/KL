#include "ConvolutionOutputWriter.h"

#include "SpikeProjector.h"

#include <fstream>
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

void WriteFilterLogLines(std::ostream &logStream, const ConvolutionConfig &config,
                         const std::vector<PhysicalFilterDescriptor> &filters,
                         const std::vector<VmaxEstimateResult> &vmaxResults)
{
    for (std::size_t filterIndex = 0; filterIndex < filters.size(); ++filterIndex) {
        const PhysicalFilterDescriptor &descriptor = filters[filterIndex];
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

} // namespace

ConvolutionOutputSession::ConvolutionOutputSession(const ConvolutionConfig &config,
                                                 const std::vector<PhysicalFilterDescriptor> &filters,
                                                 const std::vector<VmaxEstimateResult> &vmaxResults,
                                                 const std::string &configPath,
                                                 int imageCount,
                                                 int vmaxSampleCount)
    : config_(config)
    , filters_(filters)
    , vmaxResults_(vmaxResults)
    , binStream_(config.outputConvolutionsBin, std::ios::binary | std::ios::trunc)
    , csvStream_(config.outputSpikesCsv, std::ios::trunc)
    , logStream_(config.outputLog, std::ios::trunc)
{
    if (!binStream_) {
        throw std::runtime_error("cannot open output file: " + config.outputConvolutionsBin);
    }
    if (!csvStream_) {
        throw std::runtime_error("cannot open output file: " + config.outputSpikesCsv);
    }
    if (!logStream_) {
        throw std::runtime_error("cannot open output file: " + config.outputLog);
    }

    logStream_ << "version=" << kAppVersion << '\n';
    logStream_ << "config=" << configPath << '\n';
    logStream_ << "images=" << imageCount << '\n';
    logStream_ << "vmax_sample_size=" << config.vmaxSampleSize << '\n';
    logStream_ << "vmax_sample_count=" << vmaxSampleCount << '\n';
    logStream_ << "mode=" << ProjectionModeToString(config.projectionMode) << '\n';
    logStream_ << "criterion=" << config.projectionCriterion << '\n';

    for (const PhysicalFilterDescriptor &descriptor : filters_) {
        logStream_ << "map file=" << descriptor.sourceFile << " filter=" << descriptor.filterIndex
                  << " width=" << descriptor.mapSize.width << " height=" << descriptor.mapSize.height << '\n';
    }
}

void ConvolutionOutputSession::WriteImage(const BankMapsForImage &imageMaps)
{
    size_t globalFilterIndex = 0;
    bool firstCsvField = true;
    for (const FilterMapsForImage &bankMaps : imageMaps.banks) {
        for (const PolarMaps &maps : bankMaps.filters) {
            const double vmax = vmaxResults_[globalFilterIndex].vmax;
            WriteMapValues(binStream_, maps.plus);
            WriteMapValues(binStream_, maps.minus);
            WriteMapSpikes(csvStream_, maps.plus, vmax, firstCsvField);
            WriteMapSpikes(csvStream_, maps.minus, vmax, firstCsvField);
            ++globalFilterIndex;
        }
    }
    csvStream_ << '\n';
}

void ConvolutionOutputSession::Finish()
{
    WriteFilterLogLines(logStream_, config_, filters_, vmaxResults_);
}

void WriteConvolutionOutputs(const ConvolutionConfig &config,
                             const ConvolutionPipelineResult &pipeline,
                             const std::vector<VmaxEstimateResult> &vmaxResults,
                             const std::string &configPath,
                             int imageCount)
{
    ConvolutionOutputSession session(config, pipeline.filters, vmaxResults, configPath, imageCount, imageCount);
    for (const BankMapsForImage &imageMaps : pipeline.convolutions.images) {
        session.WriteImage(imageMaps);
    }
    session.Finish();
}
