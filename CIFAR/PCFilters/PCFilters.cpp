#include "PCFilters.h"

#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{

std::string Trim(const std::string &s)
{
    const std::string whitespace = " \t\r\n";
    const size_t begin = s.find_first_not_of(whitespace);
    if (begin == std::string::npos) {
        return std::string();
    }
    const size_t end = s.find_last_not_of(whitespace);
    return s.substr(begin, end - begin + 1);
}

bool StartsWith(const std::string &value, const std::string &prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

std::string ReadRequiredLine(std::ifstream &ifs, const std::string &error)
{
    std::string line;
    if (!std::getline(ifs, line)) {
        throw std::runtime_error(error);
    }
    return Trim(line);
}

void ValidateFilterMat(const cv::Mat &mat, size_t index)
{
    if (mat.empty()) {
        throw std::invalid_argument("SavePCFilters: empty filter at index " + std::to_string(index));
    }
}

double ReadValue(const cv::Mat &mat, int r, int c, int ch)
{
    switch (mat.depth()) {
    case CV_8U:
        return mat.ptr<unsigned char>(r)[c * mat.channels() + ch];
    case CV_8S:
        return mat.ptr<signed char>(r)[c * mat.channels() + ch];
    case CV_16U:
        return mat.ptr<unsigned short>(r)[c * mat.channels() + ch];
    case CV_16S:
        return mat.ptr<short>(r)[c * mat.channels() + ch];
    case CV_32S:
        return mat.ptr<int>(r)[c * mat.channels() + ch];
    case CV_32F:
        return mat.ptr<float>(r)[c * mat.channels() + ch];
    case CV_64F:
        return mat.ptr<double>(r)[c * mat.channels() + ch];
    default:
        throw std::invalid_argument("Unsupported cv::Mat depth in SavePCFilters");
    }
}

void WriteValue(cv::Mat &mat, int r, int c, int ch, double value)
{
    switch (mat.depth()) {
    case CV_8U:
        mat.ptr<unsigned char>(r)[c * mat.channels() + ch] = static_cast<unsigned char>(value);
        break;
    case CV_8S:
        mat.ptr<signed char>(r)[c * mat.channels() + ch] = static_cast<signed char>(value);
        break;
    case CV_16U:
        mat.ptr<unsigned short>(r)[c * mat.channels() + ch] = static_cast<unsigned short>(value);
        break;
    case CV_16S:
        mat.ptr<short>(r)[c * mat.channels() + ch] = static_cast<short>(value);
        break;
    case CV_32S:
        mat.ptr<int>(r)[c * mat.channels() + ch] = static_cast<int>(value);
        break;
    case CV_32F:
        mat.ptr<float>(r)[c * mat.channels() + ch] = static_cast<float>(value);
        break;
    case CV_64F:
        mat.ptr<double>(r)[c * mat.channels() + ch] = value;
        break;
    default:
        throw std::runtime_error("Unsupported cv::Mat depth in LoadPCFilters");
    }
}

void ValidateImages(const std::vector<cv::Mat> &vmat_Images, int FilterSize, int SampleSize, size_t nFilters)
{
    if (vmat_Images.empty()) {
        throw std::invalid_argument("PCFilters: vmat_Images is empty");
    }
    if (FilterSize <= 0) {
        throw std::invalid_argument("PCFilters: FilterSize must be > 0");
    }
    if (SampleSize <= 0) {
        throw std::invalid_argument("PCFilters: SampleSize must be > 0");
    }
    if (nFilters == 0U) {
        throw std::invalid_argument("PCFilters: vmat_Filters.size() must be > 0");
    }

    const cv::Size imageSize = vmat_Images.front().size();
    const int channels = vmat_Images.front().channels();
    if (channels != 1 && channels != 3) {
        throw std::invalid_argument("PCFilters: only 1 or 3 channel images are supported");
    }

    for (size_t i = 0; i < vmat_Images.size(); ++i) {
        const cv::Mat &img = vmat_Images[i];
        if (img.empty()) {
            throw std::invalid_argument("PCFilters: image is empty at index " + std::to_string(i));
        }
        if (img.size() != imageSize) {
            throw std::invalid_argument("PCFilters: image size mismatch at index " + std::to_string(i));
        }
        if (img.channels() != channels) {
            throw std::invalid_argument("PCFilters: image channel count mismatch at index " + std::to_string(i));
        }
        if (img.depth() != CV_8U) {
            throw std::invalid_argument("PCFilters: only CV_8U images are supported");
        }
        if (img.rows < FilterSize || img.cols < FilterSize) {
            throw std::invalid_argument("PCFilters: image is smaller than FilterSize at index " + std::to_string(i));
        }
    }
}

} // namespace

void PCFilters(const std::vector<cv::Mat> &vmat_Images,
               std::vector<cv::Mat> &vmat_Filters,
               int FilterSize,
               int SampleSize,
               int Seed)
{
    const size_t nFilters = vmat_Filters.size();
    ValidateImages(vmat_Images, FilterSize, SampleSize, nFilters);

    const int channels = vmat_Images.front().channels();
    const int dimension = FilterSize * FilterSize * channels;
    if (static_cast<int>(nFilters) > dimension) {
        throw std::invalid_argument("PCFilters: number of requested components is larger than patch dimension");
    }

    cv::Mat sampleMatrix(SampleSize, dimension, CV_32F);

    std::mt19937 rng;
    if (Seed >= 0) {
        rng.seed(static_cast<unsigned int>(Seed));
    } else {
        std::random_device rd;
        rng.seed(rd());
    }

    std::uniform_int_distribution<int> imageDist(0, static_cast<int>(vmat_Images.size()) - 1);
    std::uniform_int_distribution<int> xDist(0, vmat_Images.front().cols - FilterSize);
    std::uniform_int_distribution<int> yDist(0, vmat_Images.front().rows - FilterSize);

    for (int sampleIndex = 0; sampleIndex < SampleSize; ++sampleIndex) {
        const cv::Mat &src = vmat_Images[imageDist(rng)];
        const cv::Rect roi(xDist(rng), yDist(rng), FilterSize, FilterSize);
        const cv::Mat patch = src(roi);

        cv::Mat patch32f;
        patch.convertTo(patch32f, CV_MAKETYPE(CV_32F, channels));

        float *rowPtr = sampleMatrix.ptr<float>(sampleIndex);
        int outOffset = 0;
        for (int r = 0; r < FilterSize; ++r) {
            const float *patchRow = patch32f.ptr<float>(r);
            for (int c = 0; c < FilterSize; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    rowPtr[outOffset++] = patchRow[c * channels + ch];
                }
            }
        }
    }

    cv::PCA pca(sampleMatrix, cv::Mat(), cv::PCA::DATA_AS_ROW, static_cast<int>(nFilters));

    for (size_t i = 0; i < nFilters; ++i) {
        cv::Mat filter(FilterSize, FilterSize, CV_MAKETYPE(CV_32F, channels));
        const float *eigenvector = pca.eigenvectors.ptr<float>(static_cast<int>(i));

        int offset = 0;
        for (int r = 0; r < FilterSize; ++r) {
            float *row = filter.ptr<float>(r);
            for (int c = 0; c < FilterSize; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    row[c * channels + ch] = eigenvector[offset++];
                }
            }
        }

        vmat_Filters[i] = std::move(filter);
    }
}

void SavePCFilters(const std::vector<cv::Mat> &vmat_Filters, const char *pchFile)
{
    if (pchFile == nullptr || *pchFile == '\0') {
        throw std::invalid_argument("SavePCFilters: output file path is empty");
    }

    std::ofstream ofs(pchFile);
    if (!ofs) {
        throw std::runtime_error("SavePCFilters: cannot open output file");
    }

    ofs << "[PCFilters]\n";
    ofs << "version=1\n";
    ofs << "count=" << vmat_Filters.size() << "\n\n";
    ofs << std::setprecision(17);

    for (size_t i = 0; i < vmat_Filters.size(); ++i) {
        const cv::Mat &filter = vmat_Filters[i];
        ValidateFilterMat(filter, i);

        ofs << "[Filter" << i << "]\n";
        ofs << "rows=" << filter.rows << "\n";
        ofs << "cols=" << filter.cols << "\n";
        ofs << "channels=" << filter.channels() << "\n";
        ofs << "type=" << filter.type() << "\n";
        ofs << "values=";

        bool first = true;
        for (int r = 0; r < filter.rows; ++r) {
            for (int c = 0; c < filter.cols; ++c) {
                for (int ch = 0; ch < filter.channels(); ++ch) {
                    if (!first) {
                        ofs << ' ';
                    }
                    first = false;
                    ofs << ReadValue(filter, r, c, ch);
                }
            }
        }
        ofs << "\n\n";
    }
}

void LoadPCFilters(std::vector<cv::Mat> &vmat_Filters, const char *pchFile)
{
    if (pchFile == nullptr || *pchFile == '\0') {
        throw std::invalid_argument("LoadPCFilters: input file path is empty");
    }

    std::ifstream ifs(pchFile);
    if (!ifs) {
        throw std::runtime_error("LoadPCFilters: cannot open input file");
    }

    const std::string header = ReadRequiredLine(ifs, "LoadPCFilters: missing header section");
    if (header != "[PCFilters]") {
        throw std::runtime_error("LoadPCFilters: invalid header section");
    }

    const std::string versionLine = ReadRequiredLine(ifs, "LoadPCFilters: missing version");
    if (versionLine != "version=1") {
        throw std::runtime_error("LoadPCFilters: unsupported format version");
    }

    const std::string countLine = ReadRequiredLine(ifs, "LoadPCFilters: missing filter count");
    if (!StartsWith(countLine, "count=")) {
        throw std::runtime_error("LoadPCFilters: malformed count line");
    }
    const int filterCount = std::stoi(countLine.substr(std::string("count=").size()));
    if (filterCount < 0) {
        throw std::runtime_error("LoadPCFilters: negative filter count");
    }

    std::vector<cv::Mat> loadedFilters;
    loadedFilters.reserve(static_cast<size_t>(filterCount));

    std::string line;
    while (std::getline(ifs, line)) {
        line = Trim(line);
        if (line.empty()) {
            continue;
        }
        if (!StartsWith(line, "[Filter")) {
            throw std::runtime_error("LoadPCFilters: expected filter section");
        }

        const std::string rowsLine = ReadRequiredLine(ifs, "LoadPCFilters: missing rows");
        const std::string colsLine = ReadRequiredLine(ifs, "LoadPCFilters: missing cols");
        const std::string channelsLine = ReadRequiredLine(ifs, "LoadPCFilters: missing channels");
        const std::string typeLine = ReadRequiredLine(ifs, "LoadPCFilters: missing type");
        const std::string valuesLine = ReadRequiredLine(ifs, "LoadPCFilters: missing values");

        if (!StartsWith(rowsLine, "rows=") ||
            !StartsWith(colsLine, "cols=") ||
            !StartsWith(channelsLine, "channels=") ||
            !StartsWith(typeLine, "type=") ||
            !StartsWith(valuesLine, "values=")) {
            throw std::runtime_error("LoadPCFilters: malformed filter metadata");
        }

        const int rows = std::stoi(rowsLine.substr(std::string("rows=").size()));
        const int cols = std::stoi(colsLine.substr(std::string("cols=").size()));
        const int channels = std::stoi(channelsLine.substr(std::string("channels=").size()));
        const int type = std::stoi(typeLine.substr(std::string("type=").size()));

        if (rows <= 0 || cols <= 0 || channels <= 0) {
            throw std::runtime_error("LoadPCFilters: invalid filter shape");
        }
        if (CV_MAT_CN(type) != channels) {
            throw std::runtime_error("LoadPCFilters: channel count mismatch with cv::Mat type");
        }

        cv::Mat filter(rows, cols, type);
        std::istringstream valueStream(valuesLine.substr(std::string("values=").size()));

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    double value = 0.0;
                    if (!(valueStream >> value)) {
                        throw std::runtime_error("LoadPCFilters: insufficient filter values");
                    }
                    WriteValue(filter, r, c, ch, value);
                }
            }
        }

        // Ensure there are no trailing non-space characters after expected values.
        double extra = 0.0;
        if (valueStream >> extra) {
            throw std::runtime_error("LoadPCFilters: too many filter values");
        }

        loadedFilters.push_back(std::move(filter));
    }

    if (static_cast<int>(loadedFilters.size()) != filterCount) {
        throw std::runtime_error("LoadPCFilters: filter count mismatch");
    }

    vmat_Filters = std::move(loadedFilters);
}
