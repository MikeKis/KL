#include "FilterBank.h"

#include "../PCFilters/PCFilters.h"

#include <stdexcept>

void RescaleFiters(std::vector<cv::Mat> &filters_out, const std::vector<cv::Mat> &filters_in, int scale)
{
    filters_out.resize(filters_in.size());
    for (size_t i = 0; i < filters_in.size(); ++i) {
        filters_out[i] = cv::Mat(filters_in[i].rows * scale, filters_in[i].cols * scale, filters_in[i].type());
        for (int r = 0; r < filters_in[i].rows; ++r)
            for (int c = 0; c < filters_in[i].cols; ++c)
                for (int dr = 0; dr < scale; ++dr)
                    for (int dc = 0; dc < scale; ++dc)
                        std::memcpy(filters_out[i].ptr(r * scale + dr, c * scale + dc), filters_in[i].ptr(r, c), filters_in[i].elemSize());
    }
}

std::vector<LoadedFilterBank> LoadFilterBanks(const std::vector<std::string> &files, const std::vector<int> &strides, const std::vector<int> &scales)
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
        if (scales[i] == 1)
            LoadPCFilters(bank.filters, bank.file.c_str());
        else {
            std::vector<cv::Mat> filters;
            LoadPCFilters(filters, bank.file.c_str());
            RescaleFiters(bank.filters, filters, scales[i]);
        }
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
