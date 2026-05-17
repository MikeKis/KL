#include "ConvolutionRunner.h"

#include <stdexcept>

MapSize ComputeOutputMapSize(int imageWidth, int imageHeight, int filterWidth, int filterHeight, int stride)
{
    if (stride < 1) {
        throw std::invalid_argument("stride must be >= 1");
    }
    MapSize size;
    size.width = (imageWidth - filterWidth) / stride + 1;
    size.height = (imageHeight - filterHeight) / stride + 1;
    return size;
}

static double DotConvolutionAt(const cv::Mat &image, const cv::Mat &filter, int x, int y)
{
    const cv::Rect roi(x, y, filter.cols, filter.rows);
    cv::Mat patch = image(roi);
    cv::Mat patch32f;
    cv::Mat filter32f;
    if (patch.depth() != CV_32F) {
        patch.convertTo(patch32f, CV_32F);
        patch = patch32f;
    }
    if (filter.depth() != CV_32F) {
        filter.convertTo(filter32f, CV_32F);
        return filter32f.dot(patch);
    }
    return filter.dot(patch);
}

static PolarMaps ComputePolarMaps(const cv::Mat &image, const cv::Mat &filter, int stride, const MapSize &outSize)
{
    PolarMaps maps;
    maps.plus = cv::Mat(outSize.height, outSize.width, CV_64F, 0.0);
    maps.minus = cv::Mat(outSize.height, outSize.width, CV_64F, 0.0);

    for (int y = 0, yFilter = 0; y < outSize.height; ++y, yFilter += stride) {
        for (int x = 0, xFilter = 0; x < outSize.width; ++x, xFilter += stride) {
            const double value = DotConvolutionAt(image, filter, xFilter, yFilter);
            maps.plus.at<double>(y, x) = std::max(value, 0.0);
            maps.minus.at<double>(y, x) = std::max(-value, 0.0);
        }
    }
    return maps;
}

std::vector<MapSize> ComputeMapSizesForBanks(int imageWidth, int imageHeight,
                                             const std::vector<LoadedFilterBank> &banks)
{
    if (banks.empty()) {
        throw std::runtime_error("no filter banks to convolve");
    }

    std::vector<MapSize> mapSizes;
    for (const LoadedFilterBank &bank : banks) {
        const MapSize firstSize = ComputeOutputMapSize(imageWidth, imageHeight, bank.filters.front().cols,
                                                       bank.filters.front().rows, bank.stride);
        if (firstSize.width < 1 || firstSize.height < 1) {
            throw std::runtime_error("filter does not fit on image for bank: " + bank.file);
        }

        for (const cv::Mat &filter : bank.filters) {
            const MapSize size =
                ComputeOutputMapSize(imageWidth, imageHeight, filter.cols, filter.rows, bank.stride);
            if (size.width < 1 || size.height < 1) {
                throw std::runtime_error("filter does not fit on image for bank: " + bank.file);
            }
            mapSizes.push_back(size);
        }
    }
    return mapSizes;
}

BankMapsForImage ConvolveImage(const cv::Mat &image, const std::vector<LoadedFilterBank> &banks,
                               const std::vector<MapSize> &mapSizesPerBankFilter)
{
    BankMapsForImage imageMaps;
    imageMaps.banks.resize(banks.size());

    size_t globalFilterIndex = 0;
    for (size_t bankIndex = 0; bankIndex < banks.size(); ++bankIndex) {
        const LoadedFilterBank &bank = banks[bankIndex];
        FilterMapsForImage &bankMaps = imageMaps.banks[bankIndex];
        bankMaps.filters.resize(bank.filters.size());

        for (size_t filterIndex = 0; filterIndex < bank.filters.size(); ++filterIndex) {
            const MapSize &outSize = mapSizesPerBankFilter[globalFilterIndex];
            bankMaps.filters[filterIndex] =
                ComputePolarMaps(image, bank.filters[filterIndex], bank.stride, outSize);
            ++globalFilterIndex;
        }
    }
    return imageMaps;
}

ConvolutionRunResult RunConvolutions(const std::vector<cv::Mat> &images,
                                     const std::vector<LoadedFilterBank> &banks)
{
    if (images.empty()) {
        throw std::runtime_error("no images to convolve");
    }

    ConvolutionRunResult result;
    result.mapSizesPerBankFilter =
        ComputeMapSizesForBanks(images.front().cols, images.front().rows, banks);
    result.images.reserve(images.size());

    for (const cv::Mat &image : images) {
        result.images.push_back(ConvolveImage(image, banks, result.mapSizesPerBankFilter));
    }
    return result;
}
