#include "objects_arni.h"


using namespace std;

bool RasterSpikeSource::bGenerateSignals(unsigned *pfl, int bitoffset)
{
    string str;
    if (!FrameTactCounter) {
            if (!getline(cin, str).good())
                return false;
            switch (str[0]) {
            case '1': GetRasterfromText((unsigned char *)str.c_str() + 1, vuc_Raster);
                break;
            case 'T': return false;
            default:  fill(vuc_Raster.begin(), vuc_Raster.end(), 0);
                break;
            }
            FrameTactCounter = ntactperPicture;
        }
        GetSpikesfromImage(GetNReceptors(), vfl_InputSignal);
        --FrameTactCounter;


        if (dFrequencyMultiplier < 1.)
        DropSpikes(GetNReceptors(), vfl_InputSignal);
    if (rNoise)
        AddNoise(GetNReceptors(), vfl_InputSignal);
    if (Period && !(tact % Period))
        vfl_InputSignal.front() |= 1;
    if (!bitoffset)
        copy(vfl_InputSignal.begin(), vfl_InputSignal.end(), pfl);
    else copy(&vfl_InputSignal.front(), GetNReceptors(), pfl, bitoffset);
    ++tact;
    return true;
}
