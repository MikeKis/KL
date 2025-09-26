#include "objint.h"

extern NamedPipe npLearning;
extern std::unique_ptr<boost::process::child> upchi;

void FixNetwork()
{
    if (upchi) {
        upchi->terminate();
        upchi.release();
    }
    int i = -1;
    npLearning.write(i);
    npLearning.read(i);
}
