#include <boost/process/child.hpp>
#include <boost/process/pipe.hpp>
#include <boost/process/io.hpp>

#include "objint.h"

using namespace std;

void Learn(const cv::Mat &mat, int ClassId)
{
    static vector<string> vstr_Arguments = {".", "-e30000"};
    static boost::process::ipstream ips;
    static boost::process::opstream ops;
    static std::unique_ptr<boost::process::child> upchi;
    if (!upchi) {
        try {
            upchi.reset(new boost::process::child("ArNIGPU", vstr_Arguments, boost::process::std_err > ips, boost::process::std_in < ops));
            ops << '1' << strGetTextfromRaster(mat) << endl;
            ops << ClassId << endl;
        } catch (std::exception &x) {
            cout << "ArNIGPU threw the exception \"" << x.what() <<"\".\nArguments:\n";
            for (auto z: vstr_Arguments)
                cout << z << std::endl;
            exit(-1);
        } catch (...) {
            cout << "ArNIGPU threw an exception.\nArguments:\n";
            for (auto z: vstr_Arguments)
                cout << z << std::endl;
            exit(-1);
        }
    } else {
        if (ips.peek() != EOF) {
            string str;
            getline(ips, str);
            ops << '1' << strGetTextfromRaster(mat) << endl;
            ops << ClassId << endl;
        }
    }
}
