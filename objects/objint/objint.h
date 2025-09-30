#include <opencv2/opencv.hpp>
#include <boost/process/child.hpp>
#include <boost/process/pipe.hpp>
#include <boost/process/io.hpp>

#include "../objects_ArNI/objects_arni_int.h"

void Preprocess(const cv::Mat &matin, cv::Mat &matout);
void Learn(const cv::Mat &mat, int ClassId);
void FixNetwork();
void Inference(const cv::Mat &mat);
std::vector<unsigned char> vuc_fromRaster(const cv::Mat &mat);

