// This is needed to use text mode standard input for obtaining images.

#include "objint.h"

using namespace std;
using namespace cv;

boost::asio::mutable_buffer fromRaster(const cv::Mat &mat)
{
    vector<unsigned char> vuc_;
    auto y = mat.begin<double>();
    while (y != mat.end<double>()) {
        const unsigned char uc = (unsigned char)*y;
        vuc_.push_back(uc);
        ++y;
    }
    return boost::asio::buffer(vuc_);
}
