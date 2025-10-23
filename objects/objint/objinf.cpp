#include "objint.h"

using namespace std;
namespace asio = boost::asio;

NamedPipe2directional np2Inference;
std::unique_ptr<boost::process::child> upchi;

void Inference(const cv::Mat &mat)
{
    if (!upchi) {
        vector<string> vstr_Arguments = {".", "-e30001", "-f0", "-Pt", "-v1", "-r"};
        np2Inference.make(ARNI_FIFO_PATH_INFERENCE);
        upchi.reset(new boost::process::child("ArNIGPU", vstr_Arguments));
    }
    np2Inference.write(vuc_fromRaster(mat));
}
