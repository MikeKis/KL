#include <boost/asio/posix/stream_descriptor.hpp>
#include <iostream>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <boost/process/child.hpp>
#include <boost/process/pipe.hpp>
#include <boost/process/io.hpp>

#include "../objects_ArNI/objects_arni_int.h"

#include "objint.h"

using namespace std;
namespace asio = boost::asio;

void Learn(const cv::Mat &mat, int ClassId)
{
    static vector<string> vstr_Arguments = {".", "-e30000"};
    static boost::process::ipstream ips;
    static boost::process::opstream ops;
    static std::unique_ptr<boost::process::child> upchi;
    static asio::io_context io_ctx;
    static int fd;
    static asio::posix::stream_descriptor stream(io_ctx);
    if (!upchi) {
        unlink(pchArNI_fifo_path);
        if (mkfifo(pchArNI_fifo_path, 0666) == -1) {
            cerr << "Error creating FIFO: " << strerror(errno) << endl;
            exit(-1);
        }
        cout << "FIFO created at: " << pchArNI_fifo_path << endl;
        try {
            fd = open(pchArNI_fifo_path, O_RDWR | O_NONBLOCK);
            if (fd == -1) {
                cerr << "Error opening FIFO for writing: " << strerror(errno) << endl;
                unlink(pchArNI_fifo_path);
                exit(-1);
            }
            stream.assign(fd);
            upchi.reset(new boost::process::child("ArNIGPU", vstr_Arguments, boost::process::std_err > ips, boost::process::std_in < ops));
            asio::write(stream, fromRaster(mat));
            asio::write(stream, asio::buffer(&ClassId, sizeof(ClassId)));
        } catch (std::exception &x) {
            cout << "objint threw the exception \"" << x.what() <<"\nArNIGPU arguments:\n";
            for (auto z: vstr_Arguments)
                cout << z << std::endl;
            exit(-1);
        } catch (...) {
            cout << "objint threw an exception.\nArNIGPU arguments:\n";
            for (auto z: vstr_Arguments)
                cout << z << std::endl;
            exit(-1);
        }
    } else {
        int bytes_available;
        if (ioctl(fd, FIONREAD, &bytes_available) == -1) {
            cerr << "Error with ioctl(FIONREAD)\n";
            unlink(pchArNI_fifo_path);
            exit(-1);
        }
        printf("Bytes available in pipe: %d\n", bytes_available);
        if (bytes_available) {
            int i;
            asio::read(stream, asio::buffer(&i, sizeof(int)));
            asio::write(stream, fromRaster(mat));
            asio::write(stream, asio::buffer(&ClassId, sizeof(ClassId)));
        }
    }
}
