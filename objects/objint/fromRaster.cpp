// This is needed to use text mode standard input for obtaining images.

#include "objint.h"

using namespace std;
using namespace cv;

string strGetTextfromRaster(const Mat &mat)
{
    string strret;
    auto y = mat.begin<double>();
    while (y != mat.end<double>()) {
        const unsigned char uc = (unsigned char)*y;
        if (uc <= 222)
            strret += uc + 33;
        else {
            strret += ' ';
            strret += uc;
        }
        ++y;
    }
    return strret;
}
