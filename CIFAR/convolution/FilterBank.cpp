#include "FilterBank.h"

#include "../PCFilters/PCFilters.h"

#include <stdexcept>

std::vector<LoadedFilterBank> LoadFilterBanks(const std::vector<std::string> &files,
                                              const std::vector<int> &strides)
{
    if (files.size() != strides.size()) {
        throw std::invalid_argument("files and strides size mismatch");
    }

    std::vector<LoadedFilterBank> banks;
    banks.reserve(files.size());

    for (size_t i = 0; i < files.size(); ++i) {
        LoadedFilterBank bank;
        bank.file = files[i];
        bank.stride = strides[i];
        LoadPCFilters(bank.filters, bank.file.c_str());
        if (bank.filters.empty()) {
            throw std::runtime_error("filter bank is empty: " + bank.file);
        }
        banks.push_back(std::move(bank));
    }
    return banks;
}

int GetFilterChannels(const std::vector<LoadedFilterBank> &banks)
{
    if (banks.empty()) {
        throw std::runtime_error("no filter banks loaded");
    }

    const int channels = banks.front().filters.front().channels();
    for (const LoadedFilterBank &bank : banks) {
        for (const cv::Mat &filter : bank.filters) {
            if (filter.channels() != channels) {
                throw std::runtime_error("channel count mismatch across filter banks");
            }
        }
    }
    return channels;
}
