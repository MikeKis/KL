#include <fenv.h>
#include <random>
#include <map>
#include <fstream>
#include <iostream>

#include "UnsupervisedFiltersMultiple.h"

using namespace std;
using namespace cv;

// unsigned char ucGetBrightnessQuantileValue(const vector<Mat> &vmat_Images, float rBrightnessThresholdQuantile)
// {
// 	const int SampleSize = 30000;
// 	mt19937 g(300);
// 	vector<unsigned char> vuc_PixelSamples(SampleSize);
// 	for (int i = 0; i < SampleSize; ++i) {
// 		int indImage = uniform_int_distribution<int>(0, (int)(vmat_Images.size() - 1))(g);
// 		int c = uniform_int_distribution<int>(0, (int)(vmat_Images[indImage].cols - 1))(g);
// 		int r = uniform_int_distribution<int>(0, (int)(vmat_Images[indImage].rows - 1))(g);
// 		int ch = vmat_Images[indImage].channels() > 1 ? uniform_int_distribution<int>(0, (int)(vmat_Images[indImage].channels() - 1))(g) : 0;
// 		const unsigned char *ptr = vmat_Images[indImage].ptr<unsigned char>(r);
// 		vuc_PixelSamples[i] = ptr[c * vmat_Images[indImage].channels() + ch];
// 	}
// 	sort(vuc_PixelSamples.begin(), vuc_PixelSamples.end());
// 	int ind = (int)(SampleSize - SampleSize * rBrightnessThresholdQuantile);
// 	if (ind == SampleSize)
// 		--ind;
// 	return vuc_PixelSamples[ind];
// }

Mat matConvolution(const Mat &matin, const vector<Mat> &vmat_Filters, int Stride)
{
	int outwidth = (matin.cols - vmat_Filters.front().cols) / Stride + 1;
	int outheight = (matin.rows - vmat_Filters.front().rows) / Stride + 1;
	int nInputChannels = matin.channels();
    int typeConvolution = CV_64FC((int)vmat_Filters.size());
    Mat matConv(outheight, outwidth, typeConvolution, 0.);
	vector<pair<double, vector<int> > > vpdvind_;
	vector<int> vind_(3);
	int yfilter = 0;
	for (int yout = 0; yout < outheight; ++yout) {
		int xfilter = 0;
		for (int xout = 0; xout < outwidth; ++xout) {
			vector<double> vd_(vmat_Filters.size());
			for (int filter = 0; filter < vd_.size(); ++filter) {
				double d = vmat_Filters[filter].dot(Mat(matin, Rect(xfilter, yfilter, vmat_Filters.front().cols, vmat_Filters.front().rows)));
				vd_[filter] = d;
			}
			SetMultichannelPixel(matConv, yout, xout, vd_);
			xfilter += Stride;
		}
		yfilter += Stride;
	}
	return matConv;
}

Mat matMaxPoolingReLU(const Mat &matin, int Pooling, vector<double> &vd_perMapMaximum)
{
	int outwidth = matin.cols / Pooling;
	int outheight = matin.rows / Pooling;
	int nInputChannels = matin.channels();
    fill(vd_perMapMaximum.begin(), vd_perMapMaximum.end(), 0.);
    int typeout = CV_64FC(nInputChannels);
    Mat matout(outheight, outwidth, typeout);
	int yin = 0;
	for (int yout = 0; yout < outheight; ++yout) {
		int xin = 0;
		for (int xout = 0; xout < outwidth; ++xout) {
			Mat matPatch(matin, Rect(xin, yin, Pooling, Pooling));
			vector<double> vd_(nInputChannels, 0.);
			for (int r = 0; r < Pooling; ++r) {
				const double *ptr = matPatch.ptr<double>(r);
				for (int c = 0; c < Pooling; ++c)
					for (int cha = 0; cha < nInputChannels; ++cha) {
						if (*ptr > vd_[cha]) {
							vd_[cha] = *ptr;
							if (*ptr > vd_perMapMaximum[cha])
								vd_perMapMaximum[cha] = *ptr;
						}
						++ptr;
					}
			}
			SetMultichannelPixel(matout, yout, xout, vd_);
			xin += Pooling;
		}
		yin += Pooling;
	}
	return matout;
}

Mat matNormalize(const Mat &matin, vector<double> vd_perMapMaximum, int typeout, vector<float> &vr_, float rBrightnessThresholdQuantile = 0.03)
{
    Mat matout(matin.rows, matin.cols, typeout);
	int nInputChannels = matin.channels();
	for (auto &i: vd_perMapMaximum)
        if (i)
            i = 1 / i;
    vector<double> vd_;
    vd_.reserve(matin.cols * matin.rows * nInputChannels);
    for (int r = 0; r < matout.rows; ++r) {
        const auto *pin = matin.ptr<double>(r);
        auto *pout = matout.ptr<double>(r);
        for (int c = 0; c < matout.cols; ++c)
            for (int cha = 0; cha < nInputChannels; ++cha) {
                double d = *pin++ * vd_perMapMaximum[cha];
                if (d > 0.)
                    vd_.push_back(d);
                else d = 0.;
                *pout++ = d;
            }
	}
    sort(vd_.begin(), vd_.end());
    int ind = (int)(vd_.size() - vd_.size() * rBrightnessThresholdQuantile);
    if (ind == vd_.size())
        --ind;
    double dScaling = ind < vd_.size() ? 1.F / vd_[ind] : 1.F;
    for (int r = 0; r < matout.rows; ++r) {
        auto *pout = matout.ptr<double>(r);
        for (int c = 0; c < matout.cols; ++c)
            for (int cha = 0; cha < nInputChannels; ++cha) {
                *pout *= dScaling;
                if (*pout > 1.)
                    *pout = 1.;
                vr_.push_back((float)*pout);
                ++pout;
            }
    }
    return matout;
}

vector<Mat> vmat_Normalize(const vector<Mat> &vmat_Images, float rBrightnessThresholdQuantile, float &rBrightnessThreshold, float rBrightnessSaturationQuantile = 0.03)
{
    vector<Mat> vmat_ret;
    vmat_ret.reserve(vmat_Images.size());
    vector<unsigned char> vuc_;
    vector<float> vr_;
    vuc_.reserve(vmat_Images.front().cols * vmat_Images.front().rows * 3);
    for (int i = 0; i < vmat_Images.size(); ++i) {
        vuc_.clear();
        auto it = vmat_Images[i].begin<Vec3b>();
        auto it_end = vmat_Images[i].end<Vec3b>();
        for (; it != it_end; ++it) {
            const Vec3b &pixel = *it;
            vuc_.push_back(pixel[0]);
            vuc_.push_back(pixel[1]);
            vuc_.push_back(pixel[2]);
        }
        sort(vuc_.begin(), vuc_.end());
        int ind = (int)(vuc_.size() - vuc_.size() * rBrightnessSaturationQuantile);
        if (ind == vuc_.size())
            --ind;
        while (ind < vuc_.size() && !vuc_[ind])
            ++ind;
        double dScaling = ind < vuc_.size() ? 1. / vuc_[ind] : 1.;
        Mat mat(vmat_Images[i].rows, vmat_Images[i].cols, CV_64FC3);
        it = vmat_Images[i].begin<Vec3b>();
        it_end = vmat_Images[i].end<Vec3b>();
        auto itout = mat.begin<Vec3d>();
        for (; it != it_end; ++it) {
            const Vec3b &pixel = *it;
            Vec3d &pixelout = *itout++;
            pixelout[0] = min(pixel[0] * dScaling, 1.);
            vr_.push_back((float)pixelout[0]);
            pixelout[1] = min(pixel[1] * dScaling, 1.);
            vr_.push_back((float)pixelout[1]);
            pixelout[2] = min(pixel[2] * dScaling, 1.);
            vr_.push_back((float)pixelout[2]);
        }
        vmat_ret.push_back(mat);
    }
    sort(vr_.begin(), vr_.end());
    int ind = (int)(vr_.size() - vr_.size() * rBrightnessThresholdQuantile);
    rBrightnessThreshold = vr_[ind];
    return vmat_ret;
}

struct convcoo
{
    unsigned short usx;
    unsigned short usy;
    unsigned short usindFilter;
    unsigned short uspad = 0;
};

inline bool operator<(convcoo cc1, convcoo cc2){return reinterpret_cast<unsigned long long &>(cc1) < reinterpret_cast<unsigned long long &>(cc2);}

vector<vector<Mat> > vvmat_UnsupervisedFilters(const vector<Mat> &vmat_Images, const vector<ConvolutionalLayerProperties> &vclp_, float rBrightnessThresholdQuantile)
{
    // Enable trapping for division by zero
#ifdef FOR_LINUX
    if (feenableexcept(FE_DIVBYZERO) != 0) {
        perror("feenableexcept failed");
        exit(-1);
    }
#endif
    if (any_of(vclp_.begin(), vclp_.end(), [](const ConvolutionalLayerProperties &clp){return clp.nFilters > CV_CN_MAX;}))
		throw std::runtime_error("too many filters");
	if (vclp_.empty())
		throw std::runtime_error("nothing to do");
    ofstream ofslog("UnsupervisedFiltersMultiple.log.csv");
    ofstream ofsdetlog("UnsupervisedFiltersMultiple.detlog.csv");
    vector<vector<Mat> > vvmat_ret(vclp_.size());
    cout << "Normalizing images...\n";
    float rBrightnessThreshold;
    vector<Mat> vmat_CurrentTensors = vmat_Normalize(vmat_Images,rBrightnessThresholdQuantile, rBrightnessThreshold);
	mt19937 g(300);
	vector<int> vind_ran(vmat_Images.size());
    int typeInput = CV_64FC3;
    for (int Layer = 0; Layer < vvmat_ret.size(); ++Layer) {
		if (vclp_[Layer].dWmin > 0.)
			throw std::runtime_error("weight cannot be 0");
        cout << "Getting " << Layer << "-level convolutions...\nBrightness threshold " << rBrightnessThreshold << endl;
		double dw = -vclp_[Layer].dWmin * (vclp_[Layer].dWmax - vclp_[Layer].dWmin) / vclp_[Layer].dWmax;
		for (int i = 0; i < vind_ran.size(); ++i)
			vind_ran[i] = i;
		shuffle(vind_ran.begin(), vind_ran.end(), g);
		vector<Mat> vmat_FiltersW;
		vector<Mat> vmat_FiltersR;
        int nInputChannels = vmat_CurrentTensors.front().channels();
        int typeConvolutions = CV_64FC(vclp_[Layer].nFilters);
        int nInputValuesperFilter = vclp_[Layer].FilterSize * vclp_[Layer].FilterSize * nInputChannels;
        for (int i = 0; i < vclp_[Layer].nFilters; ++i) {
            vmat_FiltersW.push_back(Mat(vclp_[Layer].FilterSize, vclp_[Layer].FilterSize, typeInput));
            vmat_FiltersR.push_back(Mat(vclp_[Layer].FilterSize, vclp_[Layer].FilterSize, typeInput));
            vmat_FiltersW.back() = 0.;
            auto *pd = vmat_FiltersR.back().ptr<double>(0);
            fill(pd, pd + nInputValuesperFilter, dw);
        }
		int outwidth = (vmat_CurrentTensors.front().cols - vclp_[Layer].FilterSize) / vclp_[Layer].Stride + 1;
		int outheight = (vmat_CurrentTensors.front().rows - vclp_[Layer].FilterSize) / vclp_[Layer].Stride + 1;
        int z = 0;
        int WinnerRegionSize = 0 /* vclp_[Layer].FilterSize / 4 */;
        vector<bool> vb_SpontaneousFiring(vclp_[Layer].nFilters, false);
		for (auto ind: vind_ran) {
			const Mat &matin = vmat_CurrentTensors[ind];
            Mat matConv(outheight, outwidth, typeConvolutions, 0.);
            vector<pair<double, convcoo> > vpdcc_(2 * outheight * outwidth * vclp_[Layer].nFilters);
            convcoo cc;
            pair<double, convcoo> *ppdcc_ = &vpdcc_.front();
            pair<double, convcoo> *ppdcc_new = ppdcc_ + outheight * outwidth * vclp_[Layer].nFilters;
            int cnt = 0;
            int yfilter = 0;
            for (cc.usy = 0; cc.usy < outheight; ++cc.usy) {
				int xfilter = 0;
                for (cc.usx = 0; cc.usx < outwidth; ++cc.usx) {
					vector<double> vd_(vclp_[Layer].nFilters);
                    for (cc.usindFilter = 0; cc.usindFilter < vclp_[Layer].nFilters; ++cc.usindFilter) {
                        double d = vmat_FiltersW[cc.usindFilter].dot(Mat(matin, Rect(xfilter, yfilter, vclp_[Layer].FilterSize, vclp_[Layer].FilterSize)));
                        vd_[cc.usindFilter] = d;
                        ppdcc_[cnt++] = make_pair(d, cc);
					}
                    SetMultichannelPixel(matConv, cc.usy, cc.usx, vd_);
					xfilter += vclp_[Layer].Stride;
				}
				yfilter += vclp_[Layer].Stride;
			}
            sort(ppdcc_, ppdcc_ + cnt);
			vector<bool> vb_WeightCorrected(vclp_[Layer].nFilters, false);
            int zForced = 0;
            int zSpontaneous = 0;
            int zNoCorrection = 0;
			do {
                auto i = ppdcc_ + cnt - 1;
                while (i != ppdcc_ && (i - 1)->first == ppdcc_[cnt - 1].first)
					--i;
                if (i != ppdcc_ + cnt - 1)
                    i += uniform_int_distribution<int>(0, (int)(ppdcc_ + cnt - i - 1))(g);
                bool bModifyWeights = !vb_WeightCorrected[i->second.usindFilter];
                vb_WeightCorrected[i->second.usindFilter] = true;
				if (bModifyWeights) {
                    const Mat matWinner(matin, Rect(i->second.usx * vclp_[Layer].Stride, i->second.usy * vclp_[Layer].Stride, vclp_[Layer].FilterSize, vclp_[Layer].FilterSize));
					if (i->first < 1.) {
						int nPotentiated = 0;
						for (int r = 0; r < matWinner.rows; ++r) {
                            const auto *ptr = matWinner.ptr<double>(r);
							for (int c = 0; c < matWinner.cols * nInputChannels; ++c)
                                if (ptr[c] > rBrightnessThreshold)
									++nPotentiated;
						}
						if (nPotentiated && nPotentiated < nInputValuesperFilter) {
							double dDepression = vclp_[Layer].dLearningRate * nPotentiated / (nInputValuesperFilter - nPotentiated);
                            for (int r = 0; r < vmat_FiltersW[i->second.usindFilter].rows; ++r) {
                                double *pdW = vmat_FiltersW[i->second.usindFilter].ptr<double>(r);
                                double *pdR = vmat_FiltersR[i->second.usindFilter].ptr<double>(r);
                                const auto *ptr = matWinner.ptr<double>(r);
                                for (int c = 0; c < vclp_[Layer].FilterSize * nInputChannels; ++c) {
                                    pdR[c] += ptr[c] > rBrightnessThreshold ? vclp_[Layer].dLearningRate : -dDepression;
                                    pdW[c] = vclp_[Layer].dWmin + (vclp_[Layer].dWmax - vclp_[Layer].dWmin) * max(pdR[c], 0.) / (vclp_[Layer].dWmax - vclp_[Layer].dWmin + max(pdR[c], 0.));
                                }
							}
                            ++zForced;
                            ofsdetlog << Layer << ',' << ind << ',' << i->second.usindFilter << ",1," << nPotentiated << endl;
                        } else {
                            ++zNoCorrection;
                            ofsdetlog << Layer << ',' << ind << ',' << i->second.usindFilter << ",0," << nPotentiated << endl;
                        }
                    } else {
                        vb_SpontaneousFiring[i->second.usindFilter] = true;
                        map<double, vector<pair<int, int> > > WinnerPixels;
                        for (int r = 0; r < matWinner.rows; ++r) {
                            const auto *ptr = matWinner.ptr<double>(r);
                            for (int c = 0; c < matWinner.cols * nInputChannels; ++c)
                                WinnerPixels[ptr[c]].push_back(make_pair(r, c));
                        }
                        double dPotential = 0.;
                        auto m = WinnerPixels.rbegin();
                        int nPotentiated = 0;
                        while (dPotential < 1.) {
                            nPotentiated += (int)m->second.size();
                            for (auto n: m->second) {
                                auto oW = vmat_FiltersW[i->second.usindFilter].ptr<double>(n.first) + n.second;
                                auto oR = vmat_FiltersR[i->second.usindFilter].ptr<double>(n.first) + n.second;
                                dPotential += m->first * *oW;
                                *oR += vclp_[Layer].dLearningRate;
                                *oW = vclp_[Layer].dWmin + (vclp_[Layer].dWmax - vclp_[Layer].dWmin) * max(*oR, 0.) / (vclp_[Layer].dWmax - vclp_[Layer].dWmin + max(*oR, 0.));
                            }
                            ++m;
                        }
                        double dDepression = vclp_[Layer].dLearningRate * nPotentiated / (nInputValuesperFilter - nPotentiated);
                        while (m != WinnerPixels.rend()) {
                            for (auto n: m->second) {
                                auto oW = vmat_FiltersW[i->second.usindFilter].ptr<double>(n.first) + n.second;
                                auto oR = vmat_FiltersR[i->second.usindFilter].ptr<double>(n.first) + n.second;
                                *oR -= dDepression;
                                *oW = vclp_[Layer].dWmin + (vclp_[Layer].dWmax - vclp_[Layer].dWmin) * max(*oR, 0.) / (vclp_[Layer].dWmax - vclp_[Layer].dWmin + max(*oR, 0.));
                            }
                            ++m;
                        }
                        ++zSpontaneous;
                        ofsdetlog << Layer << ',' << ind << ',' << i->second.usindFilter << ",2," << nPotentiated << endl;
                    }
				}
                int cntnew = 0;
                for (int l = 0; l < cnt; ++l)
                    if (abs((int)i->second.usx - (int)ppdcc_[l].second.usx) > WinnerRegionSize || abs((int)i->second.usy - (int)ppdcc_[l].second.usy) > WinnerRegionSize)   // Возможно, здесь проверить не на точное совпадение, а на сильное наложение...
                        ppdcc_new[cntnew++] = ppdcc_[l];
                swap(ppdcc_, ppdcc_new);
                cnt = cntnew;
            } while (cnt && ppdcc_[cnt - 1].first >= 0. && find(vb_WeightCorrected.begin(), vb_WeightCorrected.end(), false) != vb_WeightCorrected.end());
            if (!(++z % 10000))
                cout << z << " images processed\n";
            ofslog << Layer << ',' << ind << ',' << zForced << ',' << zSpontaneous << ',' << zNoCorrection << endl;
		}
        for (int i = 0; i < vclp_[Layer].nFilters; ++i)
            if (vb_SpontaneousFiring[i]) {
                vvmat_ret[Layer].push_back(vmat_FiltersW[i]);
                ofsdetlog << Layer << ",-1," << i << ",3,-1\n";
            }
        cout << vvmat_ret[Layer].size() << " filters obtained\n";
		if (Layer == vvmat_ret.size() - 1)
			return vvmat_ret;
        typeInput = CV_64FC((int)vvmat_ret[Layer].size());
        vector<float> vr_;
        for (int tensor = 0; tensor < vmat_CurrentTensors.size(); ++tensor) {
            Mat matConv = matConvolution(vmat_CurrentTensors[tensor], vvmat_ret[Layer], vclp_[Layer].Stride);
            vector<double> vd_perMapMaximum(matConv.channels());
			Mat matPooled = matMaxPoolingReLU(matConv, vclp_[Layer].Pooling, vd_perMapMaximum);
            vmat_CurrentTensors[tensor] = matNormalize(matPooled, vd_perMapMaximum, typeInput, vr_);
		}
        sort(vr_.begin(), vr_.end());
        int ind = (int)(vr_.size() - vr_.size() * rBrightnessThresholdQuantile);
        rBrightnessThreshold = vr_[ind];
    }
	return vector<vector<Mat> >();   // never here
}

void DetermineperChannelSaturationLevel(const vector<cv::Mat> &vmat_Maps, vector<float> &vr_perChannelSaturationLevel)
{
    const int sample_size = 300000;
    mt19937 g(300);
    uniform_int_distribution<int> uidMap(0, (int)vmat_Maps.size() - 1);
    uniform_int_distribution<int> uidpos(0, (int)vmat_Maps.front().cols - 1);
    int ncha = vmat_Maps.front().channels();
    vector<vector<float> > vvr_samples(ncha, vector<float>(sample_size));
    for (int i = 0; i < sample_size; ++i) {
        const auto *ptr = vmat_Maps[uidMap(g)].ptr<float>(uidpos(g)) + uidpos(g) * ncha;
        for (auto &j: vvr_samples)
            j[i] = *ptr++;
    }
    vr_perChannelSaturationLevel.resize(ncha);
    for (int i = 0; i < ncha; ++i) {
        float rSparsity, rResultingSparsityBoost;
        vr_perChannelSaturationLevel[i] = rGetOptimumSparsitySaturationLevel10(vvr_samples[i], rSparsity, rResultingSparsityBoost);
    }
}

void Normalize(Mat &tensor, const vector<float> &vr_perChannelSaturationLevel)
{
    int ncha = tensor.channels();
    for (int r = 0; r < tensor.rows; ++r) {
        auto *ptr = tensor.ptr<float>(r);
        for (int c = 0; c < tensor.cols; ++c) 
            for (int cha = 0; cha < ncha; ++cha) {
                if (*ptr <= 0.F)
                    *ptr = 0.F;
                else if (*ptr < vr_perChannelSaturationLevel[cha])
                    *ptr /= vr_perChannelSaturationLevel[cha];
                else *ptr = 1.F;
                ++ptr;
            }
    }
}


std::vector<cv::Mat> vmat_UnsupervisedFilters(std::vector<cv::Mat> &vmat_Maps, const ConvolutionalLayerProperties &clp, float rBrightnessThresholdQuantile)
{
    // Enable trapping for division by zero
#ifdef FOR_LINUX
    static bool bDiv0Enabled = false;
    if (!bDiv0Enabled && feenableexcept(FE_DIVBYZERO) != 0) {
        perror("feenableexcept failed");
        exit(-1);
    }
    bDiv0Enabled = true;
#endif
    if (clp.nFilters > CV_CN_MAX)
        throw std::runtime_error("too many filters");
    if (clp.dWmin > 0.)
        throw std::runtime_error("weight cannot be 0");
    ofstream ofslog("UnsupervisedFilters2.log.csv");
    ofstream ofsdetlog("UnsupervisedFilters2.detlog.csv");
    vector<Mat> vmat_ret;
    int typeInput = vmat_Maps.front().type();
    cout << "Determine saturation levels...\n";

    vector<float> vr_perChannelSaturationLevel(vmat_Maps.front().channels());
    DetermineperChannelSaturationLevel(vmat_Maps, vr_perChannelSaturationLevel);

    cout << "Normalizing maps...\n";

    for (auto &tensor: vmat_Maps) 
        Normalize(tensor, vr_perChannelSaturationLevel);

    double dw = -clp.dWmin * (clp.dWmax - clp.dWmin) / clp.dWmax;
    vector<int> vind_ran(vmat_Maps.size());
    for (int i = 0; i < vind_ran.size(); ++i)
            vind_ran[i] = i;
    mt19937 g(300);
    shuffle(vind_ran.begin(), vind_ran.end(), g);
    vector<Mat> vmat_FiltersW;
    vector<Mat> vmat_FiltersR;
    int nInputChannels = vmat_Maps.front().channels();
    int typeConvolutions = CV_32FC(clp.nFilters);
    int nInputValuesperFilter = clp.FilterSize * clp.FilterSize * nInputChannels;
    for (int i = 0; i < clp.nFilters; ++i) {
        vmat_FiltersW.push_back(Mat(clp.FilterSize, clp.FilterSize, typeInput));
        vmat_FiltersR.push_back(Mat(clp.FilterSize, clp.FilterSize, typeInput));
        vmat_FiltersW.back() = 0.;
        auto *pr = vmat_FiltersR.back().ptr<float>(0);
        fill(pr, pr + nInputValuesperFilter, (float)dw);
    }
    int outwidth = (vmat_Maps.front().cols - clp.FilterSize) / clp.Stride + 1;
    int outheight = (vmat_Maps.front().rows - clp.FilterSize) / clp.Stride + 1;
    int z = 0;
    int WinnerRegionSize = 0 /* vclp_[Layer].FilterSize / 4 */;
    vector<bool> vb_SpontaneousFiring(clp.nFilters, false);
    for (auto ind: vind_ran) {
        const Mat &matin = vmat_Maps[ind];
        Mat matConv(outheight, outwidth, typeConvolutions, 0.);
        vector<pair<float, convcoo> > vprcc_(2 * outheight * outwidth * clp.nFilters);   // *2 since 2 interchanging buffers will be used
        convcoo cc;
        pair<float, convcoo> *pprcc_ = &vprcc_.front();
        pair<float, convcoo> *pprcc_new = pprcc_ + outheight * outwidth * clp.nFilters;
        int cnt = 0;
        int yfilter = 0;
        for (cc.usy = 0; cc.usy < outheight; ++cc.usy) {
            int xfilter = 0;
            for (cc.usx = 0; cc.usx < outwidth; ++cc.usx) {
                vector<float> vr_(clp.nFilters);
                for (cc.usindFilter = 0; cc.usindFilter < clp.nFilters; ++cc.usindFilter) {
                    float r = (float)vmat_FiltersW[cc.usindFilter].dot(Mat(matin, Rect(xfilter, yfilter, clp.FilterSize, clp.FilterSize)));
                    vr_[cc.usindFilter] = r;
                    pprcc_[cnt++] = make_pair(r, cc);
                }
                SetMultichannelPixel(matConv, cc.usy, cc.usx, vr_);
                xfilter += clp.Stride;
            }
            yfilter += clp.Stride;
        }
        sort(pprcc_, pprcc_ + cnt);
        vector<bool> vb_WeightCorrected(clp.nFilters, false);
        int zForced = 0;
        int zSpontaneous = 0;
        int zNoCorrection = 0;
        do {
            auto i = pprcc_ + cnt - 1;
            while (i != pprcc_ && (i - 1)->first == pprcc_[cnt - 1].first)
                --i;
            if (i != pprcc_ + cnt - 1)
                i += uniform_int_distribution<int>(0, (int)(pprcc_ + cnt - i - 1))(g);
            bool bModifyWeights = !vb_WeightCorrected[i->second.usindFilter];
            vb_WeightCorrected[i->second.usindFilter] = true;
            if (bModifyWeights) {
                const Mat matWinner(matin, Rect(i->second.usx * clp.Stride, i->second.usy * clp.Stride, clp.FilterSize, clp.FilterSize));
                if (i->first < 1.) {
                    int nPotentiated = 0;
                    for (int r = 0; r < matWinner.rows; ++r) {
                        const auto *ptr = matWinner.ptr<float>(r);
                        for (int c = 0; c < matWinner.cols * nInputChannels; ++c)
                            if (ptr[c] > rBrightnessThreshold)
                                ++nPotentiated;
                    }
                    if (nPotentiated && nPotentiated < nInputValuesperFilter) {
                        double dDepression = clp.dLearningRate * nPotentiated / (nInputValuesperFilter - nPotentiated);
                        for (int r = 0; r < vmat_FiltersW[i->second.usindFilter].rows; ++r) {
                            auto *prW = vmat_FiltersW[i->second.usindFilter].ptr<float>(r);
                            auto *prR = vmat_FiltersR[i->second.usindFilter].ptr<float>(r);
                            const auto *ptr = matWinner.ptr<float>(r);
                            for (int c = 0; c < clp.FilterSize * nInputChannels; ++c) {
                                prR[c] += (float)(ptr[c] > rBrightnessThreshold ? clp.dLearningRate : -dDepression);
                                prW[c] = (float)(clp.dWmin + (clp.dWmax - clp.dWmin) * max(prR[c], 0.F) / (clp.dWmax - clp.dWmin + max(prR[c], 0.F)));
                            }
                        }
                        ++zForced;
                    } else ++zNoCorrection;
                } else {
                    vb_SpontaneousFiring[i->second.usindFilter] = true;
                    map<double, vector<pair<int, int> > > WinnerPixels;
                    for (int r = 0; r < matWinner.rows; ++r) {
                        const auto *ptr = matWinner.ptr<float>(r);
                        for (int c = 0; c < matWinner.cols * nInputChannels; ++c)
                            WinnerPixels[ptr[c]].push_back(make_pair(r, c));
                    }
                    double dPotential = 0.;
                    auto m = WinnerPixels.rbegin();
                    int nPotentiated = 0;
                    while (dPotential < 1.) {
                        nPotentiated += (int)m->second.size();
                        for (auto n: m->second) {
                            auto oW = vmat_FiltersW[i->second.usindFilter].ptr<float>(n.first) + n.second;
                            auto oR = vmat_FiltersR[i->second.usindFilter].ptr<float>(n.first) + n.second;
                            dPotential += m->first * *oW;
                            *oR += (float)clp.dLearningRate;
                            *oW = (float)(clp.dWmin + (clp.dWmax - clp.dWmin) * max(*oR, 0.F) / (clp.dWmax - clp.dWmin + max(*oR, 0.F)));
                        }
                        ++m;
                    }
                    double dDepression = clp.dLearningRate * nPotentiated / (nInputValuesperFilter - nPotentiated);
                    while (m != WinnerPixels.rend()) {
                        for (auto n: m->second) {
                            auto oW = vmat_FiltersW[i->second.usindFilter].ptr<float>(n.first) + n.second;
                            auto oR = vmat_FiltersR[i->second.usindFilter].ptr<float>(n.first) + n.second;
                            *oR -= (float)dDepression;
                            *oW = (float)(clp.dWmin + (clp.dWmax - clp.dWmin) * max(*oR, 0.F) / (clp.dWmax - clp.dWmin + max(*oR, 0.F)));
                        }
                        ++m;
                    }
                    ++zSpontaneous;
                }
            }
            int cntnew = 0;
            for (int l = 0; l < cnt; ++l)
                if (abs((int)i->second.usx - (int)pprcc_[l].second.usx) > WinnerRegionSize || abs((int)i->second.usy - (int)pprcc_[l].second.usy) > WinnerRegionSize)   // Возможно, здесь проверить не на точное совпадение, а на сильное наложение...
                    pprcc_new[cntnew++] = pprcc_[l];
            swap(pprcc_, pprcc_new);
            cnt = cntnew;
        } while (cnt && pprcc_[cnt - 1].first >= 0. && find(vb_WeightCorrected.begin(), vb_WeightCorrected.end(), false) != vb_WeightCorrected.end());
        if (!(++z % 10000))
            cout << z << " images processed\n";
    }
    for (int i = 0; i < clp.nFilters; ++i)
        if (vb_SpontaneousFiring[i]) 
            vmat_ret.push_back(vmat_FiltersW[i]);
    cout << vmat_ret.size() << " filters obtained\n";
    return vmat_ret;
}
