#pragma once

#include <opencv2/core.hpp>

#include <vector>

void PCFilters(const std::vector<cv::Mat> &vmat_Images,
               std::vector<cv::Mat> &vmat_Filters,
               int FilterSize = 3,
               int SampleSize = 30000,
               int Seed = -1);

void SavePCFilters(const std::vector<cv::Mat> &vmat_Filters, const char *pchFile);
void LoadPCFilters(std::vector<cv::Mat> &vmat_Filters, const char *pchFile);
