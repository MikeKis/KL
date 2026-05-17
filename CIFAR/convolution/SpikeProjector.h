#pragma once

#include <cstddef>

int ProjectSpike(double value, double vmax);

double MeanSpike(const double *values, std::size_t count, double vmax);

double SaturationFraction(const double *values, std::size_t count, double vmax);
