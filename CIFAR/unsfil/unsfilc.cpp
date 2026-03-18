#include <iostream>
#include <fstream>

#include "../UnsupervisedFiltersMultiple/UnsupervisedFiltersMultiple.h"

using namespace std;
using namespace cv;

int main(int ARGC, char *ARGV[])
{
    vector<Mat> vmat_;
    ifstream ifs(ARGV[1], ios::binary);
    int n = ARGC == 4 ? atoi(ARGV[3]) : 0x7fffffff;
    int size = 32, nchannels = 3;
//    ifs.read((char *)&size, sizeof(int));
//    ifs.read((char *)&size, sizeof(int));
//    ifs.read((char *)&nchannels, sizeof(int));
    vector<unsigned char> vuc_(size * size * 3);
    for (int i = 0; i < n; ++i) {
        if (!ifs.read((char *)&vuc_.front(), size * size * nchannels).good())
            break;
        vmat_.push_back(Mat(size, size, CV_8UC3, &vuc_.front()).clone());
    }
    vector<ConvolutionalLayerProperties> vclp_ = {{3, 30, 1. / (vmat_.size() / 3), -0.3, 1.}, {4, 30, 1. / (vmat_.size() / 3), -0.3, 1.}};
    auto vmat_Filters = vvmat_UnsupervisedFilters(vmat_, vclp_);
    SaveFilters(vmat_Filters, ARGV[2]);
    std::cout << "Done!\n";
}
