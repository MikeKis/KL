#include "SpikeProjector.h"

#include <cmath>

int ProjectSpike(double value, double vmax)
{
    if (vmax <= 0.0) {
        return 0;
    }
    if (value > vmax) {
        return 10;
    }
    if (value <= 0.0) {
        return 0;
    }
    return static_cast<int>(std::round(10.0 * value / vmax));
}

double MeanSpike(const double *values, std::size_t count, double vmax)
{
    if (count == 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += static_cast<double>(ProjectSpike(values[i], vmax));
    }
    return sum / static_cast<double>(count);
}
