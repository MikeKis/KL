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
    const Rect roi(x, y, filter.cols, filter.rows);
    const Mat patch = image(roi);
    return filter.dot(patch);
}

Mat matConvolveImageReLU(const Mat &image, const vector<Mat> &filters, vector<float> &vr_Flat, int stride = 1)
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

void SaturatedSumPoolingAt(const Mat &mat, float rSaturationLevel, int PoolingSize, int maxnSpikes, int x, int y, unsigned char *pucres)
{
    const Rect roi(x, y, PoolingSize, PoolingSize);
    const Mat patch = mat(roi);
    memset(pucres, 0, mat.channels());
    for (int r = 0; r < PoolingSize; ++r) {
        const auto *pin = patch.ptr<float>(r);
        for (int c = 0; c < PoolingSize; ++c)
            for (int cha = 0; cha < mat.channels(); ++cha) {
                if (pucres[cha] < (unsigned)maxnSpikes) {
                    if (*pin >= rSaturationLevel)
                        pucres[cha] = (unsigned char)maxnSpikes;
                    else pucres[cha] += (unsigned)(maxnSpikes * *pin / rSaturationLevel);
                }
                ++pin;
            }
    }
}

Mat matSaturatedSumPooling(const Mat &mat, float rSaturationLevel, int PoolingSize = 2, int maxnSpikes = 10)
{
    int height = mat.rows / 2;
    int width = mat.cols / 2;
    int ncha = mat.channels();
    Mat matret(height, width, CV_8UC(ncha), Scalar::all(0));
    for (int y = 0, yin = 0; y < height; ++y, yin += PoolingSize) {
        auto *pout = matret.ptr<unsigned char>(y);
        for (int x = 0, xin = 0; x < width; ++x, xin += PoolingSize) {
            SaturatedSumPoolingAt(mat, rSaturationLevel, PoolingSize, maxnSpikes, xin, yin, pout);
            pout += ncha;
        }
    }
    return matret;
}

int main()
{
    vector<Mat> vmat_;
    vector<double> vd_(offsetinImage1 / sizeof(double));
    vector<float> vr_(vd_.size());
    cout << "Reading input data....\n";
    int i = 0;
    {
        ifstream ifs(pchInput, ios::binary);
        while (true) {
            ifs.seekg(nbytesperImage * i++);
            if (!ifs.read((char*)&vd_.front(), offsetinImage1).good())
                break;
            for (int j = 0; j < vd_.size(); ++j)
                vr_[j] = (float)vd_[j];
            Mat mat(MapSize0, MapSize0, CV_32FC(2 * nFiltersperScale), &vr_.front());
            vmat_.push_back(mat.clone());
        }
    }
    ifstream ifs(pchInput, ios::binary);
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
    vr_Flat.reserve(vmat_.size() * NewMapnValues0);
    for (size_t imageIndex = 0; imageIndex < vmat_.size(); ++imageIndex) 
        vmat_ConvolutionsReLU[imageIndex] = matConvolveImageReLU(vmat_[imageIndex], vvmat_Filters[0], vr_Flat);
    float rResultingSparsity, rResultingSparsityBoost;
    float rsatlev = rGetOptimumSparsitySaturationLevel10(vr_Flat, rResultingSparsity, rResultingSparsityBoost);
    vr_Flat.clear();
    vr_Flat.shrink_to_fit();
    vector<array<Mat, 3> > vamat_(vmat_.size());
    for (size_t imageIndex = 0; imageIndex < vmat_.size(); ++imageIndex)
        vamat_[imageIndex][0] = matSaturatedSumPooling(vmat_ConvolutionsReLU[imageIndex], rsatlev);
    cout << "Convolving scale 2 images....\n";
    vr_Flat.reserve(vmat_.size() * NewMapnValues1);
    for (size_t imageIndex = 0; imageIndex < vmat_2.size(); ++imageIndex)
        vmat_ConvolutionsReLU[imageIndex] = matConvolveImageReLU(vmat_2[imageIndex], vvmat_Filters[1], vr_Flat);
    rsatlev = rGetOptimumSparsitySaturationLevel10(vr_Flat, rResultingSparsity, rResultingSparsityBoost);
    for (size_t imageIndex = 0; imageIndex < vmat_.size(); ++imageIndex)
        vamat_[imageIndex][1] = matSaturatedSumPooling(vmat_ConvolutionsReLU[imageIndex], rsatlev);
    vmat_ConvolutionsReLU.clear();
    vr_Flat.clear();
    vr_Flat.shrink_to_fit();
    cout << "Converting to spikes scale 4 images....\n";
    vr_Flat.reserve(vmat_.size() * MapnValues2);
    for (size_t imageIndex = 0; imageIndex < vmat_4.size(); ++imageIndex)
        for (int r = 0; r < MapSize2; ++r) {
            const auto *pin = vmat_4[imageIndex].ptr<float>(r);
            for (int c = 0; c < MapSize2; ++c)
                for (int cha = 0; cha < 2 * nFiltersperScale; ++cha)
                    vr_Flat.push_back(*pin++);
        }
    rsatlev = rGetOptimumSparsitySaturationLevel10(vr_Flat, rResultingSparsity, rResultingSparsityBoost);
    for (size_t imageIndex = 0; imageIndex < vmat_.size(); ++imageIndex)
        vamat_[imageIndex][2] = matSaturatedSumPooling(vmat_4[imageIndex], rsatlev);
    vr_Flat.clear();
    vr_Flat.shrink_to_fit();
    ofstream ofs(pchOutput);
    bool bStarted = false;
    for (const auto &k: vamat_) {
        for (const auto &l: k)
            for (int r = 0; r < l.rows; ++r) {
                const auto *pin = l.ptr<unsigned char>(r);
                for (int c = 0; c < l.cols; ++c)
                    for (int cha = 0; cha < l.channels(); ++cha) {
                        if (!bStarted)
                            bStarted = true;
                        else ofs << ',';
                        ofs << (int)*pin++;
                    }
            }
        ofs << endl;
    }
}