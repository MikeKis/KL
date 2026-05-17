#include "../ConvolutionConfig.h"
#include "../ConvolutionPipeline.h"
#include "../ConvolutionRunner.h"
#include "../FilterBank.h"
#include "../RawImageLoader.h"
#include "../SpikeProjector.h"
#include "../VmaxEstimator.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int gFailures = 0;

void ExpectTrue(bool condition, const char *message)
{
    if (!condition) {
        ++gFailures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void ExpectNear(double actual, double expected, double tolerance, const char *message)
{
    if (std::abs(actual - expected) > tolerance) {
        ++gFailures;
        std::cerr << "FAIL: " << message << " (actual=" << actual << ", expected=" << expected << ")\n";
    }
}

void TestConfigParse()
{
    const ConvolutionConfig config = ParseConvolutionConfig("testdata/minimal.toml");
    ExpectTrue(config.inputWidth == 8, "input width");
    ExpectTrue(config.inputHeight == 8, "input height");
    ExpectTrue(config.filters.size() == 1, "filter count");
    ExpectTrue(config.filters[0].stride == 1, "stride");
    ExpectTrue(config.projectionMode == ProjectionMode::DefSparsity, "projection mode");
    ExpectTrue(config.projectionCriterion == 2.0, "criterion");
    ExpectTrue(config.vmaxSampleSize == 3000, "default vmax sample size");
}

void TestVmaxSampleIndices()
{
    const std::vector<std::size_t> all = SelectVmaxSampleIndices(10, 3000);
    ExpectTrue(all.size() == 10, "sample smaller than dataset uses all images");

    const std::vector<std::size_t> sample = SelectVmaxSampleIndices(10000, 3000);
    ExpectTrue(sample.size() == 3000, "sample capped at 3000");
    ExpectTrue(sample.front() < sample.back(), "sample indices sorted");

    bool hasDuplicate = false;
    for (std::size_t i = 1; i < sample.size(); ++i) {
        if (sample[i] == sample[i - 1]) {
            hasDuplicate = true;
        }
    }
    ExpectTrue(!hasDuplicate, "sample indices unique");
}

void TestOutputMapSize()
{
    const MapSize size33 = ComputeOutputMapSize(32, 32, 3, 3, 1);
    ExpectTrue(size33.width == 30 && size33.height == 30, "32x3x3 stride1");

    const MapSize size66 = ComputeOutputMapSize(32, 32, 6, 6, 2);
    ExpectTrue(size66.width == 14 && size66.height == 14, "32x6x6 stride2");
}

void TestReluPolarity()
{
    cv::Mat image(8, 8, CV_8UC1, cv::Scalar(100));
    cv::Mat filter(3, 3, CV_32FC1, cv::Scalar(1.0f));
    LoadedFilterBank bank;
    bank.stride = 1;
    bank.filters.push_back(filter);

    const ConvolutionRunResult result = RunConvolutions({image}, {bank});
    const PolarMaps &maps = result.images.front().banks.front().filters.front();
    ExpectTrue(maps.plus.at<double>(0, 0) > 0.0, "conv_plus positive");
    ExpectNear(maps.minus.at<double>(0, 0), 0.0, 1e-9, "conv_minus zero for positive conv");
}

void TestSpikeProjector()
{
    ExpectTrue(ProjectSpike(0.0, 5.0) == 0, "zero value");
    ExpectTrue(ProjectSpike(5.0, 5.0) == 10, "at vmax");
    ExpectTrue(ProjectSpike(6.0, 5.0) == 10, "above vmax");
    ExpectTrue(ProjectSpike(2.5, 5.0) == 5, "half vmax rounds to 5");
}

void TestVmaxEstimator()
{
    const std::vector<double> values = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    const VmaxEstimateResult sparsity = EstimateVmax(values, ProjectionMode::DefSparsity, 5.0);
    ExpectNear(sparsity.meanSpike, 5.0, 1.5, "def_sparsity mean spike");

    const VmaxEstimateResult saturation = EstimateVmax(values, ProjectionMode::DefSaturation, 0.5);
    ExpectNear(saturation.saturationFraction, 0.5, 0.2, "def_saturation fraction");
}

void TestZeroFilterError()
{
    bool threw = false;
    try {
        EstimateVmax({0.0, 0.0, 0.0}, ProjectionMode::DefSparsity, 2.0);
    } catch (const std::exception &) {
        threw = true;
    }
    ExpectTrue(threw, "all-zero filter values throw");
}

void TestZeroFilterNoOutputFiles()
{
    const std::string binPath = "testdata/should_not_exist.bin";
    const std::string csvPath = "testdata/should_not_exist.csv";
    const std::string logPath = "testdata/should_not_exist.log";
    std::remove(binPath.c_str());
    std::remove(csvPath.c_str());
    std::remove(logPath.c_str());

#ifdef _WIN32
    const int runRc = std::system("..\\..\\Workplace\\convolution.exe testdata\\zero_filter.toml");
#else
    const int runRc = std::system("../../Workplace/convolution testdata/zero_filter.toml");
#endif
    ExpectTrue(runRc != 0, "zero filter run fails");
    ExpectTrue(!std::ifstream(binPath).good(), "bin output not created");
    ExpectTrue(!std::ifstream(csvPath).good(), "csv output not created");
    ExpectTrue(!std::ifstream(logPath).good(), "log output not created");
}

std::vector<double> ReadBinaryDoubles(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("cannot open golden bin");
    }
    ifs.seekg(0, std::ios::end);
    const std::streamsize bytes = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (bytes % static_cast<std::streamsize>(sizeof(double)) != 0) {
        throw std::runtime_error("invalid golden bin size");
    }
    const std::size_t count = static_cast<std::size_t>(bytes) / sizeof(double);
    std::vector<double> values(count);
    ifs.read(reinterpret_cast<char *>(values.data()), bytes);
    return values;
}

std::string ReadTextFile(const std::string &path)
{
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("cannot open golden file");
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

static std::size_t MapValueCount(const MapSize &size)
{
    return static_cast<std::size_t>(size.width) * static_cast<std::size_t>(size.height);
}

static std::size_t ExpectedValuesPerImage(const std::vector<LoadedFilterBank> &banks, int imageWidth, int imageHeight)
{
    std::size_t total = 0;
    for (const LoadedFilterBank &bank : banks) {
        for (const cv::Mat &filter : bank.filters) {
            const MapSize mapSize =
                ComputeOutputMapSize(imageWidth, imageHeight, filter.cols, filter.rows, bank.stride);
            total += MapValueCount(mapSize) * 2U;
        }
    }
    return total;
}

static int CountCsvLines(const std::string &path)
{
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("cannot open csv: " + path);
    }
    int lines = 0;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) {
            ++lines;
        }
    }
    return lines;
}

void TestMultiFilterBanksConfig()
{
    const ConvolutionConfig config = ParseConvolutionConfig("testdata/multi_filters.toml");
    ExpectTrue(config.inputWidth == 32 && config.inputHeight == 32, "multi input size");
    ExpectTrue(config.filters.size() == 3, "three filter banks in config");
    ExpectTrue(config.filters[0].stride == 1, "3x3 stride");
    ExpectTrue(config.filters[1].stride == 2, "6x6 stride");
    ExpectTrue(config.filters[2].stride == 2, "12x12 stride");
    ExpectTrue(config.filters[0].file.find("CIFAR_3x3.PCFilters.txt") != std::string::npos,
               "3x3 workplace filter path");
    ExpectTrue(config.filters[1].file.find("CIFAR_6x6.PCFilters.txt") != std::string::npos,
               "6x6 workplace filter path");
    ExpectTrue(config.filters[2].file.find("CIFAR_12x12.PCFilters.txt") != std::string::npos,
               "12x12 workplace filter path");
}

void TestMultiFilterBanksPipeline()
{
    const ConvolutionConfig config = ParseConvolutionConfig("testdata/multi_filters.toml");

    std::vector<std::string> filterFiles;
    std::vector<int> strides;
    filterFiles.reserve(config.filters.size());
    strides.reserve(config.filters.size());
    for (const FilterConfigEntry &entry : config.filters) {
        filterFiles.push_back(entry.file);
        strides.push_back(entry.stride);
    }

    const std::vector<LoadedFilterBank> banks = LoadFilterBanks(filterFiles, strides);
    ExpectTrue(banks.size() == 3, "three banks loaded");
    ExpectTrue(banks[0].filters.size() == 20, "20 filters in 3x3 bank");
    ExpectTrue(banks[1].filters.size() == 20, "20 filters in 6x6 bank");
    ExpectTrue(banks[2].filters.size() == 20, "20 filters in 12x12 bank");
    ExpectTrue(GetFilterChannels(banks) == 3, "all banks use 3 channels");

    const std::vector<cv::Mat> images =
        LoadRawImages(config.inputFile, config.inputWidth, config.inputHeight, GetFilterChannels(banks));
    ExpectTrue(images.size() == 2, "two test images loaded");

    const ConvolutionPipelineResult pipeline = BuildPipelineResult(images, banks);
    ExpectTrue(pipeline.filters.size() == 60, "60 physical filters total");
    ExpectTrue(pipeline.convolutions.images.size() == 2, "maps for each image");

    const MapSize map33 = ComputeOutputMapSize(32, 32, 3, 3, 1);
    const MapSize map66 = ComputeOutputMapSize(32, 32, 6, 6, 2);
    const MapSize map1212 = ComputeOutputMapSize(32, 32, 12, 12, 2);
    ExpectTrue(map33.width == 30 && map33.height == 30, "3x3 output map size");
    ExpectTrue(map66.width == 14 && map66.height == 14, "6x6 output map size");
    ExpectTrue(map1212.width == 11 && map1212.height == 11, "12x12 output map size");

    ExpectTrue(pipeline.filters[0].mapSize.width == 30, "first filter map width");
    ExpectTrue(pipeline.filters[20].mapSize.width == 14, "first 6x6 filter map width");
    ExpectTrue(pipeline.filters[40].mapSize.width == 11, "first 12x12 filter map width");

    const std::size_t valuesPerImage = ExpectedValuesPerImage(banks, 32, 32);
    ExpectTrue(valuesPerImage == 48680, "values per image across all banks");
    ExpectTrue(pipeline.valuesPerFilter[0].size() == MapValueCount(map33) * 2U * images.size(),
               "per-filter value count for 3x3 bank");
    ExpectTrue(pipeline.valuesPerFilter[20].size() == MapValueCount(map66) * 2U * images.size(),
               "per-filter value count for 6x6 bank");
    ExpectTrue(pipeline.valuesPerFilter[40].size() == MapValueCount(map1212) * 2U * images.size(),
               "per-filter value count for 12x12 bank");
}

void TestMultiFilterBanksIntegration()
{
#ifdef _WIN32
    const int runRc = std::system("..\\..\\Workplace\\convolution.exe testdata\\multi_filters.toml");
#else
    const int runRc = std::system("../../Workplace/convolution testdata/multi_filters.toml");
#endif
    ExpectTrue(runRc == 0, "multi-filter convolution run");

    const std::size_t expectedDoubles = 48680U * 2U;
    const std::vector<double> binValues = ReadBinaryDoubles("testdata/out_multi_conv_values.bin");
    ExpectTrue(binValues.size() == expectedDoubles, "multi-filter bin value count");

    const int csvLines = CountCsvLines("testdata/out_multi_conv_spikes.csv");
    ExpectTrue(csvLines == 2, "csv row count equals image count");

    const std::string logText = ReadTextFile("testdata/out_multi_convolution.log");
    ExpectTrue(logText.find("CIFAR_3x3.PCFilters.txt") != std::string::npos, "log mentions 3x3 filters");
    ExpectTrue(logText.find("CIFAR_6x6.PCFilters.txt") != std::string::npos, "log mentions 6x6 filters");
    ExpectTrue(logText.find("CIFAR_12x12.PCFilters.txt") != std::string::npos, "log mentions 12x12 filters");
    ExpectTrue(logText.find("filter=19") != std::string::npos, "log includes last filter index");
}

void TestGoldenIntegration()
{
#ifdef _WIN32
    const int runRc = std::system("..\\..\\Workplace\\convolution.exe testdata\\minimal.toml");
#else
    const int runRc = std::system("../../Workplace/convolution testdata/minimal.toml");
#endif
    ExpectTrue(runRc == 0, "convolution run for golden test");

    const std::vector<double> expectedBin = ReadBinaryDoubles("testdata/expected_conv_values.bin");
    const std::vector<double> actualBin = ReadBinaryDoubles("testdata/out_conv_values.bin");
    ExpectTrue(expectedBin == actualBin, "golden bin match");

    const std::string expectedCsv = ReadTextFile("testdata/expected_conv_spikes.csv");
    const std::string actualCsv = ReadTextFile("testdata/out_conv_spikes.csv");
    ExpectTrue(expectedCsv == actualCsv, "golden csv match");
}

} // namespace

int main()
{
    try {
        TestConfigParse();
        TestVmaxSampleIndices();
        TestOutputMapSize();
        TestReluPolarity();
        TestSpikeProjector();
        TestVmaxEstimator();
        TestZeroFilterError();
        TestZeroFilterNoOutputFiles();
        TestMultiFilterBanksConfig();
        TestMultiFilterBanksPipeline();
        TestMultiFilterBanksIntegration();
        TestGoldenIntegration();
    } catch (const std::exception &ex) {
        std::cerr << "convolution_test error: " << ex.what() << '\n';
        return 1;
    }

    if (gFailures > 0) {
        std::cerr << gFailures << " test failure(s)\n";
        return 1;
    }

    std::cout << "All convolution tests passed\n";
    return 0;
}
