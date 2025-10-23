#include <iostream>

#include "objects_arni.h"

using namespace std;

extern LabelSpikeSource *plssG;
extern bool bCommandCopyNetwork;
extern NamedPipe2directional np2;

RasterSpikeSource::RasterSpikeSource(bool bLearning, int nReceptors): vuc_Raster(nReceptors)
{
    SetNReceptors(nReceptors);
    np2.open(bLearning ? ARNI_FIFO_PATH_LEARNING : ARNI_FIFO_PATH_INFERENCE);
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
    if (!(plssG->tact % RecordPresentationPeriod)) {
        int i = 0;
        if (np2.strName() == ARNI_FIFO_PATH_LEARNING)
            np2.read(i);
        if (i >= 0) {
            plssG->CurrentClass = i;
            size_t bytes_read = np2.read(vuc_Raster);
            if (bytes_read != GetNReceptors()) {
                cout << "Error reading FIFO\n";
                exit(-40000);
            }
        } else {
            fill(vuc_Raster.begin(), vuc_Raster.end(), 0);
            bCommandCopyNetwork = true;
        }
    }
    if (plssG->tact % RecordPresentationPeriod < PicturePresentationTime)
        GetSpikesfromImage(pfl);
    return true;
}
