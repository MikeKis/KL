#include "VmaxEstimator.h"

#include "SpikeProjector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

double SaturationFraction(const std::vector<double> &values, double vmax)
{
    std::size_t above = 0;
    for (double value : values) {
        if (value > vmax) {
            ++above;
        }
    }
    return static_cast<double>(above) / static_cast<double>(values.size());
}

std::vector<double> UniqueCandidates(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

} // namespace

VmaxEstimateResult EstimateVmax(const std::vector<double> &values, ProjectionMode mode, double criterion)
{
    if (values.empty()) {
        throw std::runtime_error("cannot estimate Vmax for empty value set");
    }

    const double maxValue = *std::max_element(values.begin(), values.end());
    if (maxValue <= 0.0) {
        throw std::runtime_error("all convolution values are zero for a filter; cannot estimate Vmax");
    }

    const std::vector<double> candidates = UniqueCandidates(values);
    VmaxEstimateResult best;
    best.vmax = candidates.back();
    double bestError = std::numeric_limits<double>::infinity();

    for (double candidate : candidates) {
        if (candidate <= 0.0) {
            continue;
        }

        double error = 0.0;
        double meanSpike = 0.0;
        double saturationFraction = 0.0;

        if (mode == ProjectionMode::DefSparsity) {
            meanSpike = MeanSpike(values.data(), values.size(), candidate);
            saturationFraction = SaturationFraction(values, candidate);
            error = std::abs(meanSpike - criterion);
        } else {
            saturationFraction = SaturationFraction(values, candidate);
            meanSpike = MeanSpike(values.data(), values.size(), candidate);
            error = std::abs(saturationFraction - criterion);
        }

        if (error < bestError) {
            bestError = error;
            best.vmax = candidate;
            best.meanSpike = meanSpike;
            best.saturationFraction = saturationFraction;
        }
    }

    if (best.vmax <= 0.0) {
        throw std::runtime_error("failed to find positive Vmax candidate");
    }

    // Recompute metrics at the chosen Vmax for consistency.
    best.meanSpike = MeanSpike(values.data(), values.size(), best.vmax);
    best.saturationFraction = SaturationFraction(values, best.vmax);
    return best;
}
