#include "VmaxEstimator.h"

#include "SpikeProjector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace std;

const double dSparsityRelativeTolerance = 0.03;

double dSaturationLevelfromSparsity(const std::vector<double> &vd_sorted, double dTargetMeanFiringFrequency)
{
    double d = vd_sorted[vd_sorted.size() / 2];
    double dhigh = d;
    double dlow = d;
    double dcri = MeanSpike(&vd_sorted.front(), vd_sorted.size(), d);
    if (dcri < dTargetMeanFiringFrequency) {
        double dlim = *upper_bound(vd_sorted.begin(), vd_sorted.end(), 0.);
        do {
            dlow *= 0.5;
        } while (d > dlim && MeanSpike(&vd_sorted.front(), vd_sorted.size(), d) < dTargetMeanFiringFrequency);
        if (d <= dlim)
            throw std::runtime_error("sparsity criterion cannot be satisfied");
    } else do {
        dhigh *= 2;
    } while (MeanSpike(&vd_sorted.front(), vd_sorted.size(), d) > dTargetMeanFiringFrequency);
    const double dSparsityAbsoluteTolerance = dSparsityRelativeTolerance * dTargetMeanFiringFrequency;
    do {
        d = (dlow + dhigh) / 2;
        dcri = MeanSpike(&vd_sorted.front(), vd_sorted.size(), d);
        if (abs(dcri - dTargetMeanFiringFrequency) < dSparsityAbsoluteTolerance)
            break;
        if (dcri > dTargetMeanFiringFrequency)
            dlow = d;
        else dhigh = d;
    } while (true);
    return d;
}

VmaxEstimateResult EstimateVmax(const std::vector<double> &values, ProjectionMode mode, double criterion)
{

    if (values.empty()) {
        throw std::runtime_error("cannot estimate Vmax for empty value set");
    }

    auto vd_ = values;
    std::sort(vd_.begin(), vd_.end());
    VmaxEstimateResult best;
    if (mode == ProjectionMode::DefSparsity) {
        best.vmax = dSaturationLevelfromSparsity(vd_, criterion);
        best.saturationFraction = (vd_.end() - std::lower_bound(vd_.begin(), vd_.end(), best.vmax)) / (double)vd_.size();
    } else {
        int ind = (int)(vd_.size() * (1 - criterion));
        best.vmax = ind < vd_.size() ? vd_[ind] : vd_.back();
    }
    return best;
}
