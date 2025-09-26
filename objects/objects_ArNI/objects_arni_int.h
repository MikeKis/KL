#ifndef OBJECTS_ARNI_INT_H
#define OBJECTS_ARNI_INT_H

#define ARNI_FIFO_PATH_LEARNING "/tmp/object_ArNI_learning"
#define ARNI_FIFO_PATH_INFERENCE "/tmp/object_ArNI_inference"

#include <iostream>

#include <boost/asio.hpp>
#include <boost/asio/posix/stream_descriptor.hpp>

class NamedPipe
{
    boost::asio::io_context               io_ctx;
    int                                   fd;
    boost::asio::posix::stream_descriptor stream;
public:
    NamedPipe(): fd(-1), stream(io_ctx) {}
    void open(const char *pchname)
    {
        fd = ::open(pchname, O_RDWR);
        if (fd == -1) {
            std::cerr << "Error opening FIFO for reading: " << strerror(errno) << std::endl;
            exit(-40000);
        }
        stream.assign(fd);
        strname = pchname;
    }
    template<class T> void write(const T &a){boost::asio::write(stream, boost::asio::buffer(&a, sizeof(a)));}
    template<class T> void write(const std::vector<T> &v_){boost::asio::write(stream, boost::asio::buffer(v_));}
    template<class T> void write(boost::asio::mutable_buffer buf){boost::asio::write(stream, buf);}
    template<class T> void read(T &a)
    {
        size_t bytes_read = boost::asio::read(stream, boost::asio::buffer(&a, sizeof(a)));
        if (bytes_read != sizeof(a)) {
            std::cerr << "Error reading FIFO\n";
            exit(-40000);
        }
    }
    template<class T> size_t read(std::vector<T> &v_){return boost::asio::read(stream, boost::asio::buffer(v_));}
    int DataAvailable() const
    {
        int bytes_available;
        if (ioctl(fd, FIONREAD, &bytes_available) == -1) {
            std::cerr << "Error with ioctl(FIONREAD)\n";
            unlink(strname.c_str());
            exit(-1);
        }
        return bytes_available;
    }
    ~NamedPipe(){stream.close();}
    std::string strname;
};

#endif // OBJECTS_ARNI_INT_H
