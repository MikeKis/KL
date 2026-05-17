#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

std::vector<cv::Mat> LoadRawImages(const std::string &filePath, int width, int height, int channels);
