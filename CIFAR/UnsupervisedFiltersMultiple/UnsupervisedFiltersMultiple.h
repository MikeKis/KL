#pragma once

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write
#include <opencv2/videoio.hpp> 

struct ConvolutionalLayerProperties
{
	int FilterSize;
	int nFilters;
	double dLearningRate;
	double dWmin;
	double dWmax;
	int Stride = 1;
	int Pooling = 2;
};

std::vector<std::vector<cv::Mat> > vvmat_UnsupervisedFilters(const std::vector<cv::Mat> &vmat_Images, const std::vector<ConvolutionalLayerProperties> &vclp_, float rBrightnessThresholdQuantile = 0.25F);
std::vector<cv::Mat> vmat_UnsupervisedFilters(std::vector<cv::Mat> &vmat_Maps, const ConvolutionalLayerProperties &clp, float rBrightnessThresholdQuantile = 0.25F);

template<class T> void SetMultichannelPixel(cv::Mat &mat, int r, int c, const std::vector<T> &v_)
{
	T *prow = mat.ptr<T>(r);
	int offset = c * mat.channels();
	for (int k = 0; k < mat.channels(); ++k)
		prow[offset + k] = v_[k];
}

float rGetOptimumSparsitySaturationLevel10(const std::vector<float> &vr_, float &rResultingSparsity, float &rResultingSparsityBoost);
const int maxnSpikesPerObject = 10;
const float rBrightnessThreshold = 1.F / maxnSpikesPerObject;

template<class T> void SaveFilters(const std::vector<std::vector<cv::Mat> > &vvmat_Filters, const char *pchFile)
{
	std::ofstream ofs(pchFile);
	int Layer = 0;
	for (const auto &j: vvmat_Filters) {
        ofs << "*** LAYER " << Layer++ << std::endl;
		for (const auto &i: j) {
            for (int r = 0; r < i.rows; ++r) {
                const auto *pin = i.ptr<T>(r);
                for (int c = 0; c < i.cols; ++c) {
                    if (c)
                        ofs << ',';
                    ofs << '(';
                    for (int cha = 0; cha < i.channels(); ++cha) {
                        if (cha)
                            ofs << ',';
                        ofs << *pin++;
                    }
                    ofs << ')';
                }
                ofs << std::endl;
            }
			ofs << std::endl;
		}
	}
}

template<class T> cv::Mat matctor(int size, int nchannels);

template<class T> void LoadFilters(std::vector<std::vector<cv::Mat> > &vvmat_Filters, const char *pchFile)
{
    vvmat_Filters.clear();
    std::ifstream ifs(pchFile);
    std::string s;
    if (!std::getline(ifs, s).good()) {
        std::cout << "Unexpected end of input file!\n";
        exit(-1);
    }
    do {
        vvmat_Filters.push_back(std::vector<cv::Mat>());
        int FilterSize = 0;
        int nChannels = 0;
        while (std::getline(ifs, s).good() && s[0] != '*') {
            if (vvmat_Filters.back().empty()) {
                FilterSize = (int)std::count(s.begin(), s.end(), '(');
                nChannels = (int)std::count(s.begin(), s.begin() + s.find(')'), ',') + 1;
            }
            vvmat_Filters.back().push_back(matctor<T>(FilterSize, nChannels));
            auto &i = vvmat_Filters.back().back();
            for (int r = 0; r < FilterSize; ++r) {
                auto *pout = i.ptr<T>(r);
                if (s.empty()) {
                    if (!std::getline(ifs, s).good()) {
                        std::cout << "Unexpected end of input file!\n";
                        exit(-1);
                    }
                }
                int n = FilterSize * nChannels;
                size_t pos = 0;
                for (int j = 0; j < n; ++j) {
                    pos = s.find_first_not_of("(,)\n", pos);
                    size_t posend = s.find_first_of("(,)\n", pos);
                    *pout++ = (T)atof(s.substr(pos, posend - pos).c_str());
                    pos = posend;
                }
                s.clear();
            }
            if (!std::getline(ifs, s).good()) {   // empty line
                std::cout << "Unexpected end of input file!\n";
                exit(-1);
            }
        }
    } while (ifs.good());
}
