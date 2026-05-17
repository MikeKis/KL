#include "../ConvolutionConfig.h"
#include "../ConvolutionRunner.h"
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
    const int runRc = std::system("..\\..\\x64\\Debug\\convolution.exe testdata\\zero_filter.toml");
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

void TestGoldenIntegration()
{
#ifdef _WIN32
    const int runRc = std::system("..\\..\\x64\\Debug\\convolution.exe testdata\\minimal.toml");
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
        TestOutputMapSize();
        TestReluPolarity();
        TestSpikeProjector();
        TestVmaxEstimator();
        TestZeroFilterError();
        TestZeroFilterNoOutputFiles();
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
