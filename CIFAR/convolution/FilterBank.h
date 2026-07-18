#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct LoadedFilterBank {
    std::string file;
    int stride = 1;
    std::vector<cv::Mat> filters;
};

std::vector<LoadedFilterBank> LoadFilterBanks(const std::vector<std::string> &files, const std::vector<int> &strides,
                                              const std::vector<int> &scales);

inline std::vector<LoadedFilterBank> LoadFilterBanks(const std::vector<std::string> &files,
                                                     const std::vector<int> &strides)
{
    return LoadFilterBanks(files, strides, std::vector<int>(files.size(), 1));
}

int GetFilterChannels(const std::vector<LoadedFilterBank> &banks);
