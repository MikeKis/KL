#pragma once

#include <string>
#include <vector>

enum class ProjectionMode {
    DefSparsity,
    DefSaturation,
};

struct FilterConfigEntry {
    std::string file;
    int         stride = 1;
    int         scale = 1;
};

struct ConvolutionConfig {
    std::string inputFile;
    int inputWidth = 0;
    int inputHeight = 0;
    std::vector<FilterConfigEntry> filters;
    ProjectionMode projectionMode = ProjectionMode::DefSparsity;
    double projectionCriterion = 0.0;
    int vmaxSampleSize = 3000;
    std::string outputConvolutionsBin;
    std::string outputSpikesCsv;
    std::string outputLog;
};

const char *ProjectionModeToString(ProjectionMode mode);

ConvolutionConfig ParseConvolutionConfig(const char *configPath);
