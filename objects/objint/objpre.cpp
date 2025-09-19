#include "objint.h"

using namespace std;
using namespace cv;

const int stddev_filter_size = 3;
const int stddev_stride = 1;
const int output_object_size = 31;

double ddist2(Vec3b v3b1, Vec3b v3b2)
{
    double d = (int)(unsigned)v3b2[0] - (int)(unsigned)v3b1[0];
    double dret = d * d;
    d = (int)(unsigned)v3b2[1] - (int)(unsigned)v3b1[1];
    dret += d * d;
    d = (int)(unsigned)v3b2[2] - (int)(unsigned)v3b1[2];
    dret += d * d;
    return dret;
}

double dstd(const Mat &mat)
{
    int indCenter = mat.cols / 2;
    auto Center = mat.at<Vec3b>(indCenter, indCenter);

    // Calculate spatial differences

    double d = ddist2(Center, mat.at<Vec3b>(0, indCenter));
    d += ddist2(Center, mat.at<Vec3b>(mat.cols - 1, indCenter));
    d += ddist2(Center, mat.at<Vec3b>(indCenter, 0));
    d += ddist2(Center, mat.at<Vec3b>(indCenter, mat.cols - 1));

    return sqrt(d);
}

Mat matStdDevTransformation(const Mat &matin)
{
    Mat stddev((matin.rows - stddev_filter_size) / stddev_stride + 1, (matin.cols - stddev_filter_size) / stddev_stride + 1, CV_64F);

    int yfilter = 0;

    for (int iy = 0; iy < stddev.rows; ++iy) {
        int xfilter = 0;
        for (int ix = 0; ix < stddev.cols; ++ix) {
            Mat mat = cv::Mat(matin, Rect(xfilter, yfilter, stddev_filter_size, stddev_filter_size));
            stddev.at<double>(iy, ix) = dstd(mat);
            xfilter += stddev_stride;
        }
        yfilter += stddev_stride;
    }

    // Normalize output values to the range [0, 255] - like monochrome image.

    double d, dmax;
    minMaxIdx(stddev, &d, &dmax);

    double r = 255 / dmax;
    stddev *= r;

    return stddev;
}

inline double dmaximum(const cv::Mat& mat)
{
    double d, dmax;
    cv::minMaxIdx(mat, &d, &dmax);
    return dmax;
}

inline double dfloor(double d)
{
    double d1 = floor(d);
    return d == d1 ? d - 1 : d1;
}

template<class T> void Shrink(const cv::Mat &matin, double dScale, T(*fn)(const cv::Mat &), cv::Mat& matout)
{
    matout = cv::Mat((int)round(matin.rows / dScale), (int)round(matin.cols / dScale), matin.type());

    double dstepx = (double)matin.cols / matout.cols;
    double dstepy = (double)matin.rows / matout.rows;
    double dyin = -0.5;
    for (int iyout = 0; iyout < matout.rows; ++iyout, dyin += dstepy) {
        int y = (int)ceil(dyin);
        int ysize = (int)dfloor(dyin + dstepy) - y + 1;
        double dxin = -0.5;
        for (int ixout = 0; ixout < matout.cols; ++ixout, dxin += dstepx) {
            int x = (int)ceil(dxin);
            int xsize = (int)dfloor(dxin + dstepx) - x + 1;
            double d, dmax;
            cv::minMaxIdx(cv::Mat(matin, cv::Rect(x, y, xsize, ysize)), &d, &dmax);
            matout.at<double>(iyout, ixout) = dmax;
        }
    }
}

template<class T> void Shrink(const cv::Mat &matin, int Scale, T(*fn)(const cv::Mat &), cv::Mat &matout)
{
    matout = cv::Mat(matin.rows / Scale, matin.cols / Scale, matin.type());

    int yfilter = 0;

    for (int iy = 0; iy < matout.rows; ++iy) {
        int xfilter = 0;
        for (int ix = 0; ix < matout.cols; ++ix) {
            matout.at<T>(iy, ix) = fn(cv::Mat(matin, cv::Rect(xfilter, yfilter, Scale, Scale)));
            xfilter += Scale;
        }
        yfilter += Scale;
    }

}

void Preprocess(const Mat &matin, Mat &matout)
{
    if (matin.cols != matin.rows)
        throw std::runtime_error("non-square image");
    Mat stddev = matStdDevTransformation(matin);
    if (stddev.cols % output_object_size)
        Shrink<double>(stddev, (double)stddev.cols / output_object_size, dmaximum, matout);
    else Shrink<double>(stddev, stddev.cols / output_object_size, dmaximum, matout);
}
