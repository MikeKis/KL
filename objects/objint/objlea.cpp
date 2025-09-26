#include "objint.h"

using namespace std;

NamedPipe npLearning;

void Learn(const cv::Mat &mat, int ClassId)
{
    static vector<string> vstr_Arguments = {".", "-e30000"};
    static std::unique_ptr<boost::process::child> upchi;
    if (!upchi) {
        unlink(ARNI_FIFO_PATH_LEARNING);
        if (mkfifo(ARNI_FIFO_PATH_LEARNING, 0666) == -1) {
            cerr << "Error creating FIFO: " << strerror(errno) << endl;
            exit(-1);
        }
        cout << "FIFO created at: " << ARNI_FIFO_PATH_LEARNING << endl;
        npLearning.open(ARNI_FIFO_PATH_LEARNING);
        upchi.reset(new boost::process::child("ArNIGPU", vstr_Arguments));
    }
    npLearning.write(ClassId);
    npLearning.write(fromRaster(mat));
}
