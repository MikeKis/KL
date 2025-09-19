#include <opencv2/opencv.hpp>

void Preprocess(const cv::Mat &matin, cv::Mat &matout);
void Learn(const cv::Mat &mat, int ClassId);
std::string strGetTextfromRaster(const cv::Mat &mat);

