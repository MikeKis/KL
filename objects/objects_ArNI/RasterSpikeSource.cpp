#include <iostream>

#include "objects_arni.h"

using namespace std;
namespace asio = boost::asio;

extern LabelSpikeSource *plssG;

RasterSpikeSource::RasterSpikeSource()
{
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
        size_t bytes_read = asio::read(plssG->stream, asio::buffer(vuc_Raster));
        if (bytes_read != GetNReceptors()) {
            cout << "Error reading FIFO\n";
            exit(-40000);
        }
        bytes_read = asio::read(plssG->stream, asio::buffer(&plssG->CurrentClass, sizeof(plssG->CurrentClass)));
        if (bytes_read != sizeof(plssG->CurrentClass)) {
            cout << "Error reading FIFO\n";
            exit(-40000);
        }
        FrameTactCounter = RecordPresentationPeriod;
    }
    if (FrameTactCounter-- > RecordPresentationPeriod - PicturePresentationTime)
        GetSpikesfromImage(pfl);
    return true;
}
