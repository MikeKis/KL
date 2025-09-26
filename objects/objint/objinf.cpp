#include "objint.h"

using namespace std;
namespace asio = boost::asio;

NamedPipe npInference;
std::unique_ptr<boost::process::child> upchi;

void Inference(const cv::Mat &mat)
{
    if (!upchi) {
        vector<string> vstr_Arguments = {".", "-e30001", "-f0"};
        unlink(ARNI_FIFO_PATH_INFERENCE);
        if (mkfifo(ARNI_FIFO_PATH_INFERENCE, 0666) == -1) {
            cerr << "Error creating FIFO: " << strerror(errno) << endl;
            exit(-1);
        }
        cout << "FIFO created at: " << ARNI_FIFO_PATH_INFERENCE << endl;
        npInference.open(ARNI_FIFO_PATH_INFERENCE);
        upchi.reset(new boost::process::child("ArNIGPU", vstr_Arguments));
    }
    npInference.write(fromRaster(mat));
}
