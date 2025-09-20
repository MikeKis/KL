#include <iostream>

#include "objects_arni.h"

using namespace std;

LabelSpikeSource *plssG = NULL;

RECEPTORS_SET_PARAMETERS(pchMyReceptorSectionName, nReceptors, xn)
{
    if (!strcmp(pchMyReceptorSectionName, "R")) {
        plssG = new LabelSpikeSource;
        return new RasterSpikeSource;
    } else if (!strcmp(pchMyReceptorSectionName, "Target"))
        return plssG;
    else {
        cout << "objects_ArNI -- unknown section name: " << pchMyReceptorSectionName << endl;
        exit(-1);
    }
}

DYNAMIC_LIBRARY_ENTRY_POINT IReceptors *LoadStatus(Serializer &ser) {return new LabelSpikeSource;}   // just nothing
READOUT_SET_PARAMETERS(ExperimentId, tactTermination, nOutputNeurons, xn){plssG->vn_PredictionVotes.resize(nOutputNeurons, 0);}

READOUT_OBTAIN_SPIKES(v_Firing)
{
    plssG->ObtainOutputSpikes(v_Firing);
    return true;
}

READOUT_FINALIZE(OriginalTerminationCode, filGenLog) {return 0;}
DYNAMIC_LIBRARY_ENTRY_POINT void Serialize(Serializer &ser, bool bSave) {}
