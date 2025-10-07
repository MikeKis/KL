#include <iostream>

#include "objects_arni.h"

using namespace std;

LabelSpikeSource *plssG = NULL;
bool bCommandCopyNetwork = false;

RECEPTORS_SET_PARAMETERS(pchMyReceptorSectionName, nReceptors, xn)
{
    if (!strcmp(pchMyReceptorSectionName, "R")) {
        bool bLearning = !strcmp(xn.child_value("mode"), "learning");
        plssG = new LabelSpikeSource(bLearning);
        return new RasterSpikeSource(bLearning, nReceptors);
    }
    if (!strcmp(pchMyReceptorSectionName, "Target")) {
        plssG->SetNReceptors(nReceptors);
        return plssG;
    }
    cout << "objects_ArNI -- unknown section name: " << pchMyReceptorSectionName << endl;
    exit(-1);
}

DYNAMIC_LIBRARY_ENTRY_POINT IReceptors *LoadStatus(Serializer &ser) {return new LabelSpikeSource(false);}   // just nothing
NETWORK_SET_PARAMETERS(xn, NetworkCopy, pchSectionPrefix){}

NETWORK_MODIFY(CurrentTact)
{
    if (!bCommandCopyNetwork)
        return false;
    inc.ScheduleSavingNetwork(CurrentTact + 1, "obj.F.nns");
    bCommandCopyNetwork = false;
    return false;
}

READOUT_SET_PARAMETERS(ExperimentId, tactTermination, nOutputNeurons, xn){plssG->vn_PredictionVotes.resize(nOutputNeurons, 0);}

READOUT_OBTAIN_SPIKES(v_Firing)
{
    plssG->ObtainOutputSpikes(v_Firing);
    return true;
}

READOUT_FINALIZE(OriginalTerminationCode, filGenLog)
{
    return 0;
}
DYNAMIC_LIBRARY_ENTRY_POINT void Serialize(Serializer &ser, bool bSave) {}
