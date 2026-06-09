#include <fstream>
#include <iostream>

#include "UnsupervisedFiltersMultiple.h"

using namespace std;

const float qstep = 0.0001F;

float rGetOptimumSparsitySaturationLevel10(const std::vector<float> &vr_, float &rResultingSparsity, float &rResultingSparsityBoost)
{
	static vector<float> vr_NormalSparsity10;
	if (vr_NormalSparsity10.empty()) {
		ifstream ifsNormalSparsity10("NormalSparsity10.csv");
		string s;
		getline(ifsNormalSparsity10, s);
		while (getline(ifsNormalSparsity10, s).good()) 
			vr_NormalSparsity10.push_back((float)atof(s.substr(s.find(',') + 1).c_str()));
		if (vr_NormalSparsity10.empty()) {
			cout << "NormalSparsity10.csv not found!\n";
			exit(-1);
		}
	}
	auto vr_sorted = vr_;
	sort(vr_sorted.begin(), vr_sorted.end());
	int indpos = (int)(upper_bound(vr_sorted.begin(), vr_sorted.end(), 0.F) - vr_sorted.begin());
	rResultingSparsityBoost = 0;
	int indthr;
	for (int i = 0; i < vr_NormalSparsity10.size(); ++i) {
		int ind = (int)(vr_sorted.size() * (1.F - qstep * (i + 1)));
		float rvstep = vr_sorted[ind] / maxnSpikesPerObject;
		size_t nSpikes = maxnSpikesPerObject * (vr_sorted.size() - ind);
		int nSpikesperObject = 0;
		float thrval = rvstep;
		for (int j = indpos; j < ind; ++j) {
			if (vr_sorted[j] >= thrval) {
				nSpikesperObject = (int)(vr_sorted[j] / rvstep);
				thrval = (nSpikesperObject + 1) * rvstep;
			}
			nSpikes += nSpikesperObject;
		}
		float rSparsity = (float)nSpikes / (maxnSpikesPerObject * vr_sorted.size());
		float rSparsityBoost = vr_NormalSparsity10[i] / rSparsity;
		if (rSparsityBoost > rResultingSparsityBoost) {
			rResultingSparsityBoost = rSparsityBoost;
			rResultingSparsity = rSparsity;
			indthr = ind;
		}
	}
	return vr_sorted[indthr];
}
