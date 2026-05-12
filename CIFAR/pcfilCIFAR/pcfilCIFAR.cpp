#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>

#include "../PCFilters/PCFilters.h"

int main(int argc, char *argv[])
{
    const char *inputFile = argc > 1 ? argv[1] : "..\\Workplace\\CIFAR10.bin";
    const char *filters3File = argc > 2 ? argv[2] : "..\\Workplace\\PCFilters_3x3.txt";
    const char *filters6File = argc > 3 ? argv[3] : "..\\Workplace\\PCFilters_6x6.txt";
    const char *filters12File = argc > 4 ? argv[4] : "..\\Workplace\\PCFilters_12x12.txt";

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

        std::vector<cv::Mat> filters3(kFilterCount);
        std::vector<cv::Mat> filters6(kFilterCount);
        std::vector<cv::Mat> filters12(kFilterCount);

        PCFilters(images, filters3, 3, kSampleSize, 300);
        PCFilters(images, filters6, 6, kSampleSize, 600);
        PCFilters(images, filters12, 12, kSampleSize, 1200);

        SavePCFilters(filters3, filters3File);
        SavePCFilters(filters6, filters6File);
        SavePCFilters(filters12, filters12File);

        std::cout << "Saved 3x3 filters to: " << filters3File << "\n";
        std::cout << "Saved 6x6 filters to: " << filters6File << "\n";
        std::cout << "Saved 12x12 filters to: " << filters12File << "\n";
        std::cout << "Done!\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "pcfilCIFAR error: " << ex.what() << "\n";
        return 1;
    }
}
