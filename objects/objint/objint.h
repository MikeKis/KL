#include <opencv2/opencv.hpp>
#include <boost/asio.hpp>

void Preprocess(const cv::Mat &matin, cv::Mat &matout);
void Learn(const cv::Mat &mat, int ClassId);
void FixNetwork();
int Inference(const cv::Mat &mat);
boost::asio::mutable_buffer fromRaster(const cv::Mat &mat);

