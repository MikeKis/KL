#include <iostream>
#include <fstream>
#include <vector>

#include "../UnsupervisedFiltersMultiple/UnsupervisedFiltersMultiple.h"

using namespace std;
using namespace cv;

const char *pchInput = "CIFAR_1_2_4_scaled.bin";   // It contains non-negative convolutuions (after ReLU) with PCA filters (2 * filter count)
const char *pchFilters = "CIFAR_1_2_scaled.filters.txt";
const char *pchOutput = "CIFAR_2_level.nspikes.csv";
const int OriginalImageSize = 32;
const int nInputChannels = 3;
const int nFiltersperScale = 20;
const int FilterSize = 3;
const int nImagesUsed = 50000;
const int MapSize0 = OriginalImageSize - FilterSize + 1;
const int MapSize1 = OriginalImageSize / 2 - FilterSize + 1;
const int MapSize2 = OriginalImageSize / 4 - FilterSize + 1;
const int offsetinImage1 = 2 * nFiltersperScale * MapSize0 * MapSize0 * sizeof(double);
const int offsetinImage2 = offsetinImage1 + 2 * nFiltersperScale * MapSize1 * MapSize1 * sizeof(double);
const int MapnValues2 = 2 * nFiltersperScale * MapSize2 * MapSize2;
const size_t nbytesperImage = offsetinImage2 + MapnValues2 * sizeof(double);

double DotConvolutionAt(const Mat &image, const Mat &filter, int x, int y)
{
    const cv::Rect roi(x, y, filter.cols, filter.rows);
    cv::Mat patch = image(roi);
    return filter.dot(patch);
}

Mat ConvolveImageReLU(const Mat &image, const vector<Mat> &filters, vector<float> &vr_Flat, int stride = 1)
{
    int height = (image.rows - filters.front().rows) / stride + 1;
    int width = (image.cols - filters.front().cols) / stride + 1;
    Mat matret(height, width, CV_32FC((int)filters.size()), 0.0F);

    for (int y = 0, yFilter = 0; y < height; ++y, yFilter += stride) {
        auto *pout = matret.ptr<float>(y);
        for (int x = 0, xFilter = 0; x < width; ++x, xFilter += stride)
            for (const auto &i: filters) {
                const float value = (float)DotConvolutionAt(image, i, xFilter, yFilter);
                *pout++ = std::max(value, 0.0F);
                vr_Flat.push_back(value);
            }
    }
    return matret;
}

int main()
{
    ifstream ifs(pchInput, ios::binary);
    vector<Mat> vmat_;
    vector<double> vd_(offsetinImage1 / sizeof(double));
    vector<float> vr_(vd_.size());
    cout << "Reading input data....\n";
    int i = 0;
    while (true) {
        ifs.seekg(nbytesperImage * i++);
        if (!ifs.read((char*)&vd_.front(), offsetinImage1).good())
            break;
        for (int j = 0; j < vd_.size(); ++j)
            vr_[j] = (float)vd_[j];
        Mat mat(MapSize0, MapSize0, CV_32FC(2 * nFiltersperScale), &vr_.front());
        vmat_.push_back(mat.clone());
    }
    int nImagesUsed = (int)vmat_.size();
    vector<Mat> vmat_2(nImagesUsed);
    vd_.resize((offsetinImage2 - offsetinImage1) / sizeof(double));
    vr_.resize(vd_.size());
    cout << "Reading input data - scale 2....\n";
    for (int i = 0; i < nImagesUsed; ++i) {
        ifs.seekg(nbytesperImage * i + offsetinImage1);
        if (!ifs.read((char*)&vr_.front(), offsetinImage2 - offsetinImage1).good()) {
            cout << "Unexpected end of input file!\n";
            exit(-1);
        }
        for (int j = 0; j < vd_.size(); ++j)
            vr_[j] = (float)vd_[j];
        Mat mat(MapSize1, MapSize1, CV_32FC(2 * nFiltersperScale), &vr_.front());
        vmat_2[i] = mat.clone();
    }
    vector<Mat> vmat_4(nImagesUsed);
    vd_.resize((nbytesperImage - offsetinImage2) / sizeof(double));
    vr_.resize(vd_.size());
    cout << "Reading input data - scale 4....\n";
    for (int i = 0; i < nImagesUsed; ++i) {
        ifs.seekg(nbytesperImage * i + offsetinImage2);
        if (!ifs.read((char*)&vr_.front(), nbytesperImage - offsetinImage2).good()) {
            cout << "Unexpected end of input file!\n";
            exit(-1);
        }
        for (int j = 0; j < vd_.size(); ++j)
            vr_[j] = (float)vd_[j];
        Mat mat(MapSize2, MapSize2, CV_32FC(2 * nFiltersperScale), &vr_.front());
        vmat_4[i] = mat.clone();
    }
    cout << "Reading filters....\n";
    vector<vector<Mat> > vvmat_Filters;
    LoadFilters<float>(vvmat_Filters, pchFilters);
    vector<Mat> vmat_ConvolutionsReLU(vmat_.size());
    cout << "Convolving unscaled images....\n";
    vector<float> vr_Flat;
    int NewMapSize0 = MapSize0 - vvmat_Filters[0].front().rows + 1;
    int NewMapSize1 = MapSize1 - vvmat_Filters[1].front().rows + 1;
    int NewMapnValues0 = (int)(NewMapSize0 * NewMapSize0 * vvmat_Filters[0].size());
    int NewMapnValues1 = (int)(NewMapSize1 * NewMapSize1 * vvmat_Filters[1].size());
    vr_Flat.reserve(vmat_.size() * NewMapSize0 * NewMapSize0 * vvmat_Filters[0].size());
    for (size_t imageIndex = 0; imageIndex < vmat_.size(); ++imageIndex)
        vmat_ConvolutionsReLU[imageIndex] = ConvolveImageReLU(vmat_[imageIndex], vvmat_Filters[0], vr_Flat);
    float rResultingSparsity, rResultingSparsityBoost;
    float rsatlev = rGetOptimumSparsitySaturationLevel10(vr_Flat, rResultingSparsity, rResultingSparsityBoost);
    vector<vector<unsigned char> > vvuc_(vmat_.size(), vector<unsigned char>(NewMapnValues0 + NewMapnValues1 + MapnValues2));
    i = 0;
    for (size_t imageIndex = 0; imageIndex < vmat_.size(); ++imageIndex) 
        for (int j = 0; j < NewMapnValues0; ++j) 
            vvuc_[imageIndex][j] = (unsigned char)(10 * min(1.F, vr_Flat[i++] / rsatlev));
    vr_Flat.clear();
    cout << "Convolving scale 2 images....\n";
    vr_Flat.reserve(vmat_.size() * NewMapSize1 * NewMapSize1 * vvmat_Filters[1].size());
    for (size_t imageIndex = 0; imageIndex < vmat_2.size(); ++imageIndex)
        vmat_ConvolutionsReLU[imageIndex] = ConvolveImageReLU(vmat_2[imageIndex], vvmat_Filters[1], vr_Flat);
    vmat_ConvolutionsReLU.clear();
    rsatlev = rGetOptimumSparsitySaturationLevel10(vr_Flat, rResultingSparsity, rResultingSparsityBoost);
    i = 0;
    for (size_t imageIndex = 0; imageIndex < vmat_2.size(); ++imageIndex)
        for (int j = 0; j < NewMapnValues1; ++j)
            vvuc_[imageIndex][NewMapnValues0 + j] = (unsigned char)(10 * min(1.F, vr_Flat[i++] / rsatlev));
    vr_Flat.clear();
    cout << "Converting to spikes scale 4 images....\n";
    vr_Flat.reserve(vmat_4.size() * MapnValues2);
    for (size_t imageIndex = 0; imageIndex < vmat_4.size(); ++imageIndex)
        for (int r = 0; r < MapSize2; ++r) {
            const auto *pin = vmat_4[imageIndex].ptr<float>(r);
            for (int c = 0; c < MapSize2; ++c)
                for (int cha = 0; cha < 2 * nFiltersperScale; ++cha)
                    vr_Flat.push_back(*pin++);
        }
    rsatlev = rGetOptimumSparsitySaturationLevel10(vr_Flat, rResultingSparsity, rResultingSparsityBoost);
    i = 0;
    for (size_t imageIndex = 0; imageIndex < vmat_4.size(); ++imageIndex)
        for (int j = 0; j < MapnValues2; ++j)
            vvuc_[imageIndex][NewMapnValues0 + NewMapnValues1 + j] = (unsigned char)(10 * min(1.F, vr_Flat[i++] / rsatlev));
    vr_Flat.clear();
    ofstream ofs(pchOutput);
    for (const auto &k: vvuc_) 
        for (int j = 0; j < k.size(); ++j)
            ofs << (int)k[j] << (j < k.size() - 1 ? ',' : '\n');
}