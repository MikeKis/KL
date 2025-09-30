#include "objint.h"

extern NamedPipe2directional np2Learning;
extern std::unique_ptr<boost::process::child> upchi;

void FixNetwork()
{
    if (upchi) {
        upchi->terminate();
        upchi.release();
    }
    int i = -1;
    np2Learning.write(i);
    np2Learning.read(i);
}
