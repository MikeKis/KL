#include "ConvolutionConfig.h"

#include <toml++/toml.hpp>

#include <stdexcept>

const char *ProjectionModeToString(ProjectionMode mode)
{
    switch (mode) {
    case ProjectionMode::DefSparsity:
        return "def_sparsity";
    case ProjectionMode::DefSaturation:
        return "def_saturation";
    }
    return "unknown";
}

static ProjectionMode ParseProjectionMode(const std::string &mode)
{
    if (mode == "def_sparsity") {
        return ProjectionMode::DefSparsity;
    }
    if (mode == "def_saturation") {
        return ProjectionMode::DefSaturation;
    }
    throw std::runtime_error("projection.mode must be def_sparsity or def_saturation");
}

static void ValidateCriterion(ProjectionMode mode, double criterion)
{
    if (mode == ProjectionMode::DefSparsity) {
        if (criterion < 0.0 || criterion > 10.0) {
            throw std::runtime_error("projection.criterion for def_sparsity must be in [0, 10]");
        }
        return;
    }
    if (criterion <= 0.0 || criterion > 1.0) {
        throw std::runtime_error("projection.criterion for def_saturation must be in (0, 1]");
    }
}

ConvolutionConfig ParseConvolutionConfig(const char *configPath)
{
    if (configPath == nullptr || *configPath == '\0') {
        throw std::runtime_error("config path is empty");
    }

    const toml::table root = toml::parse_file(configPath);
    ConvolutionConfig config;

    const toml::table *input = root["input"].as_table();
    if (input == nullptr) {
        throw std::runtime_error("missing [input] section");
    }
    config.inputFile = input->get("file")->value_or("");
    config.inputWidth = static_cast<int>(input->get("width")->value_or(0));
    config.inputHeight = static_cast<int>(input->get("height")->value_or(0));
    if (config.inputFile.empty() || config.inputWidth <= 0 || config.inputHeight <= 0) {
        throw std::runtime_error("input.file, input.width, and input.height are required");
    }

    const toml::array *filters = root["filters"].as_array();
    if (filters == nullptr || filters->empty()) {
        throw std::runtime_error("filters array must be non-empty");
    }
    for (const toml::node &node : *filters) {
        const toml::table *entry = node.as_table();
        if (entry == nullptr) {
            throw std::runtime_error("filters entries must be tables");
        }
        FilterConfigEntry filterEntry;
        filterEntry.file = entry->get("file")->value_or("");
        filterEntry.stride = static_cast<int>(entry->get("stride")->value_or(0));
        if (filterEntry.file.empty()) {
            throw std::runtime_error("filters[].file is required");
        }
        if (filterEntry.stride < 1) {
            throw std::runtime_error("filters[].stride must be >= 1");
        }
        config.filters.push_back(std::move(filterEntry));
    }

    const toml::table *projection = root["projection"].as_table();
    if (projection == nullptr) {
        throw std::runtime_error("missing [projection] section");
    }
    const std::string mode = projection->get("mode")->value_or("");
    config.projectionMode = ParseProjectionMode(mode);
    config.projectionCriterion = projection->get("criterion")->value_or(0.0);
    if (const toml::node *vmaxSampleNode = projection->get("vmax_sample_size")) {
        config.vmaxSampleSize = static_cast<int>(vmaxSampleNode->value_or(3000));
    }
    ValidateCriterion(config.projectionMode, config.projectionCriterion);
    if (config.vmaxSampleSize <= 0) {
        throw std::runtime_error("projection.vmax_sample_size must be positive");
    }

    const toml::table *output = root["output"].as_table();
    if (output == nullptr) {
        throw std::runtime_error("missing [output] section");
    }
    config.outputConvolutionsBin = output->get("convolutions_bin")->value_or("");
    config.outputSpikesCsv = output->get("spikes_csv")->value_or("");
    config.outputLog = output->get("log")->value_or("");
    if (config.outputConvolutionsBin.empty() || config.outputSpikesCsv.empty() || config.outputLog.empty()) {
        throw std::runtime_error("output.convolutions_bin, output.spikes_csv, and output.log are required");
    }

    return config;
}
