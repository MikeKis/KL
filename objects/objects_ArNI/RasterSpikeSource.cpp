#include <iostream>

#include "objects_arni_int.h"
#include "objects_arni.h"

using namespace std;
namespace asio = boost::asio;

extern int CurrentClass;

RasterSpikeSource::RasterSpikeSource(): stream(io_ctx)
{
    fd = open(fifo_path, O_RDWR);
    if (fd == -1) {
        cout << "Error opening FIFO for reading: " << strerror(errno) << std::endl;
        exit(-40000);
    }
    stream.assign(fd);
    vd_State.resize(GetNReceptors(), 0.);
    dStateIncrementFactor = dmaxFrequency / 255;
}

bool RasterSpikeSource::bGenerateSignals(unsigned *pfl, int bitoffset)
{
    if (bitoffset) {
        cout << "Non-zero bitoffset\n";
        exit(-40000);
    }
    fill(pfl, pfl + (GetNReceptors() - 1) / 32 + 1, 0);
    if (!FrameTactCounter) {
        size_t bytes_read = asio::read(stream, asio::buffer(vuc_Raster));
        if (bytes_read != GetNReceptors()) {
            cout << "Error reading FIFO\n";
            exit(-40000);
        }
        bytes_read = asio::read(stream, asio::buffer(&CurrentClass, sizeof(CurrentClass)));
        if (bytes_read != sizeof(CurrentClass)) {
            cout << "Error reading FIFO\n";
            exit(-40000);
        }
        FrameTactCounter = RecordPresentationPeriod;
    }
    if (FrameTactCounter-- > RecordPresentationPeriod - PicturePresentationTime)
        GetSpikesfromImage(pfl);
    return true;
}
