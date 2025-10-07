#include "objint.h"

using namespace std;

NamedPipe2directional np2Learning;

void Learn(const cv::Mat &mat, int ClassId)
{
    static vector<string> vstr_Arguments = {".", "-e30000", "-Pt"};
    static std::unique_ptr<boost::process::child> upchi;
    if (!upchi) {
        np2Learning.make(ARNI_FIFO_PATH_LEARNING);
        upchi.reset(new boost::process::child("ArNIGPU", vstr_Arguments));
        cout << "ArNIGPU started with argumetns . -e30000\n";
    }
    np2Learning.write(ClassId);
    auto vuc_ = vuc_fromRaster(mat);
    np2Learning.write(vuc_);
    cout << "Frame sent to ArNIGPU (class " << ClassId << ")\n";
}
