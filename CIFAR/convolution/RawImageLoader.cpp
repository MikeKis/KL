#include "RawImageLoader.h"

#include <fstream>
#include <stdexcept>

std::vector<cv::Mat> LoadRawImages(const std::string &filePath, int width, int height, int channels)
{
    if (width <= 0 || height <= 0 || channels <= 0) {
        throw std::invalid_argument("invalid image dimensions");
    }

    std::ifstream ifs(filePath, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("cannot open input file: " + filePath);
    }

    const std::streamsize fileSize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    const std::size_t imageBytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                                   static_cast<std::size_t>(channels);
    if (fileSize < 0 || static_cast<std::size_t>(fileSize) % imageBytes != 0) {
        throw std::runtime_error("input file size is not a multiple of image size");
    }

    const std::size_t imageCount = static_cast<std::size_t>(fileSize) / imageBytes;
    if (imageCount == 0) {
        throw std::runtime_error("no images in input file");
    }

    const int matType = CV_MAKETYPE(CV_8U, channels);
    std::vector<unsigned char> raw(imageBytes);
    std::vector<cv::Mat> images;
    images.reserve(imageCount);

    for (std::size_t i = 0; i < imageCount; ++i) {
        if (!ifs.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(raw.size()))) {
            throw std::runtime_error("unexpected end of input file while reading images");
        }
        images.push_back(cv::Mat(height, width, matType, raw.data()).clone());
    }

    return images;
}
