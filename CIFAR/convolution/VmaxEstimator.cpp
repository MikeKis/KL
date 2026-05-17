#include "VmaxEstimator.h"

#include "SpikeProjector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

static std::vector<double> UniquePositiveCandidates(const std::vector<double> &values)
{
    std::vector<double> candidates;
    candidates.reserve(values.size());
    for (double value : values) {
        if (value > 0.0) {
            candidates.push_back(value);
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    if (candidates.empty()) {
        candidates.push_back(1.0);
    }
    return candidates;
}

VmaxEstimateResult EstimateVmax(const std::vector<double> &values, ProjectionMode mode, double criterion)
{
    if (values.empty()) {
        throw std::runtime_error("cannot estimate Vmax for empty value set");
    }

    const double maxValue = *std::max_element(values.begin(), values.end());
    if (maxValue <= 0.0) {
        throw std::runtime_error("all convolution values are zero for filter");
    }

    const std::vector<double> candidates = UniquePositiveCandidates(values);
    VmaxEstimateResult best;
    double bestError = std::numeric_limits<double>::infinity();

    for (double vmax : candidates) {
        VmaxEstimateResult trial;
        trial.vmax = vmax;
        trial.meanSpike = MeanSpike(values.data(), values.size(), vmax);
        trial.saturationFraction = SaturationFraction(values.data(), values.size(), vmax);

        double error = 0.0;
        if (mode == ProjectionMode::DefSparsity) {
            error = std::abs(trial.meanSpike - criterion);
        } else {
            error = std::abs(trial.saturationFraction - criterion);
        }

        if (error < bestError || (error == bestError && vmax < best.vmax)) {
            bestError = error;
            best = trial;
        }
    }

    return best;
}
