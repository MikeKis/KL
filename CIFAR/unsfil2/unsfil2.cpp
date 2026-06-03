#include <iostream>
#include <fstream>

#include "../UnsupervisedFiltersMultiple/UnsupervisedFiltersMultiple.h"

using namespace std;
using namespace cv;

const char *pchInput = "CIFAR_1_2_4_scaled.bin";
const char *pchOutput = "CIFAR_1_2_scaled.filters.txt";
const int OriginalImageSize = 32;
const int nInputChannels = 3;
const int nFiltersperScale = 20;
const int FilterSize = 3;
const int nImagesUsed = 50000;
const int MapSize0 = OriginalImageSize - FilterSize + 1;
const int MapSize1 = OriginalImageSize / 2 - FilterSize + 1;
const int MapSize2 = OriginalImageSize / 4 - FilterSize + 1;
const int offsetinImage1 = 2 * nFiltersperScale * MapSize0 * MapSize0 * sizeof(float);
const int offsetinImage2 = offsetinImage1 + 2 * nFiltersperScale * MapSize1 * MapSize1 * sizeof(float);
const size_t nbytesperImage = offsetinImage2 + 2 * nFiltersperScale * MapSize2 * MapSize2 * sizeof(float);

int main(int ARGC, char* ARGV[])
{
    ifstream ifs(pchInput, ios::binary);
    vector<Mat> vmat_(nImagesUsed);
    vector<float> vr_(offsetinImage1 / sizeof(float));
    for (int i = 0; i < nImagesUsed; ++i) {
        ifs.seekg(nbytesperImage * i);
        if (!ifs.read((char*)&vr_.front(), offsetinImage1).good()) {
            cout << "Unexpected end of input file!\n";
            exit(-1);
        }
        vmat_.push_back(Mat(MapSize0, MapSize0, CV_32FC(2 * nFiltersperScale), &vr_.front()).clone());
    }
    ConvolutionalLayerProperties clp{4, 30, 1. / (vmat_.size() / 3), -0.3, 1.};
    auto vmat_Filters0 = vmat_UnsupervisedFilters(vmat_, clp);   // vmat_ will be normalized in place
    vr_.resize((offsetinImage2 - offsetinImage1) / sizeof(float));
    for (int i = 0; i < nImagesUsed; ++i) {
        ifs.seekg(nbytesperImage * i + offsetinImage1);
        if (!ifs.read((char*)&vr_.front(), offsetinImage2 - offsetinImage1).good()) {
            cout << "Unexpected end of input file!\n";
            exit(-1);
        }
        vmat_.push_back(Mat(MapSize1, MapSize1, CV_32FC(2 * nFiltersperScale), &vr_.front()).clone());
    }
    auto vmat_Filters1 = vmat_UnsupervisedFilters(vmat_, clp);   // vmat_ will be normalized in place
    SaveFilters(vector<vector<cv::Mat> >{vmat_Filters0, vmat_Filters1}, pchOutput);
    std::cout << "Done!\n";
}
