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
    if (!upchi) {
        unlink(fifo_path);
        if (mkfifo(fifo_path, 0666) == -1) {
            cerr << "Error creating FIFO: " << strerror(errno) << endl;
            exit(-1);
        }
        cout << "FIFO created at: " << fifo_path << endl;
        try {
            int fd = open(fifo_path, O_RDWR | O_NONBLOCK);
            if (fd == -1) {
                cerr << "Error opening FIFO for writing: " << strerror(errno) << endl;
                unlink(fifo_path);
                exit(-1);
            }
            asio::posix::stream_descriptor stream(io_ctx, fd);
            upchi.reset(new boost::process::child("ArNIGPU", vstr_Arguments, boost::process::std_err > ips, boost::process::std_in < ops));
            asio::write(stream, fromRaster(mat));
            asio::write(stream, asio::buffer(&ClassId, sizeof(ClassId)));
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



int main() {

    // Create named pipe (FIFO)

    // Remove existing FIFO if it exists

    // Create new FIFO with read/write permissions
    if (mkfifo(fifo_path, 0666) == -1) {
        std::cerr << "Error creating FIFO: " << strerror(errno) << std::endl;
        return 1;
    }

    std::cout << "FIFO created at: " << fifo_path << std::endl;

    try {
        // Open the FIFO for writing (non-blocking to avoid hanging)
        int fd = open(fifo_path, O_WRONLY | O_NONBLOCK);
        if (fd == -1) {
            std::cerr << "Error opening FIFO for writing: " << strerror(errno) << std::endl;
            unlink(fifo_path);
            return 1;
        }

        // Create stream descriptor from file descriptor
        asio::posix::stream_descriptor stream(io_ctx, fd);

        // Message to send
        std::string message = "Hello from Boost.Asio named pipe!";

        // Write message to the FIFO
        asio::write(stream, asio::buffer(message));

        std::cout << "Message sent: " << message << std::endl;

        // Close the stream
        stream.close();

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        unlink(fifo_path);
        return 1;
    }

    // Clean up
    unlink(fifo_path);

    return 0;
}
