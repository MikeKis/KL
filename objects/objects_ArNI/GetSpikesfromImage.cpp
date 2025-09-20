#include "objects_arni.h"

using namespace std;

void RasterSpikeSource::GetSpikesfromImage(unsigned *pfl)
{
    BitMaskAccess bma;
    for (int l = 0; l < vuc_Raster.size(); ++l, ++bma) {
        vd_State[l] += dStateIncrementFactor * vuc_Raster[l];
        if (vd_State[l] >= 1.) {
            pfl |= bma;
            --vd_State[l];
        }
    }
}
