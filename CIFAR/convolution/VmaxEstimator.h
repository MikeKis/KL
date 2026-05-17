#pragma once

#include "ConvolutionConfig.h"

#include <string>
#include <vector>

struct VmaxEstimateResult {
    double vmax = 0.0;
    double meanSpike = 0.0;
    double saturationFraction = 0.0;
};

VmaxEstimateResult EstimateVmax(const std::vector<double> &values,
                                ProjectionMode mode,
                                double criterion);
