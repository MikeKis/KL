#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "../PCFilters/PCFilters.h"

int main(int argc, char *argv[])
{
    const char *inputFile = argc > 1 ? argv[1] : "..\\Workplace\\CIFAR10.bin";
    const char *filters2File = argc > 2 ? argv[2] : "..\\Workplace\\PCFilters_scale_2.txt";
    const char *filters4File = argc > 3 ? argv[3] : "..\\Workplace\\PCFilters_scale_4.txt";

    constexpr int kImageSize = 32;
    constexpr int kChannels = 3;
    constexpr int kMaxImages = 50000;
    constexpr int kSampleSize = 30000;
    constexpr int kFilterCount = 20;

    try {
        std::ifstream ifs(inputFile, std::ios::binary);
        if (!ifs) {
            throw std::runtime_error(std::string("Cannot open input file: ") + inputFile);
        }

        std::vector<cv::Mat> images;
        images.reserve(kMaxImages);
        std::vector<unsigned char> raw(kImageSize * kImageSize * kChannels);

        // Keep CIFAR parsing identical to unsfil/unsfilc.cpp: raw consecutive images without labels.
        for (int i = 0; i < kMaxImages; ++i) {
            if (!ifs.read(reinterpret_cast<char *>(&raw.front()), static_cast<std::streamsize>(raw.size())).good()) {
                break;
            }
            images.push_back(cv::Mat(kImageSize, kImageSize, CV_8UC3, &raw.front()).clone());
        }

        if (images.empty()) {
            throw std::runtime_error("No images loaded from input file");
        }

        std::cout << "Loaded images: " << images.size() << "\n";

        std::vector<cv::Mat> imagesPooled2(kMaxImages);
        std::vector<cv::Mat> imagesPooled4(kMaxImages);
        for (int i = 0; i < kMaxImages; ++i) {
            cv::boxFilter(images[i], imagesPooled2[i], -1, cv::Size(2, 2), cv::Point(-1, -1), true);
            cv::boxFilter(images[i], imagesPooled4[i], -1, cv::Size(4, 4), cv::Point(-1, -1), true);
        }

        std::vector<cv::Mat> filters2(kFilterCount);
        std::vector<cv::Mat> filters4(kFilterCount);

        PCFilters(imagesPooled2, filters2, 3, kSampleSize, 300);
        PCFilters(imagesPooled4, filters4, 3, kSampleSize, 600);

        SavePCFilters(filters2, filters2File);
        SavePCFilters(filters4, filters4File);

        std::cout << "Saved scale 2 filters to: " << filters2File << "\n";
        std::cout << "Saved scale 4 filters to: " << filters4File << "\n";
        std::cout << "Done!\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "pcfilCIFAR error: " << ex.what() << "\n";
        return 1;
    }
}
