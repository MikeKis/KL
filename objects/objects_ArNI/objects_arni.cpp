#include <iostream>

#include "objects_arni.h"

using namespace std;

RasterSpikeSource *prssG = NULL;

RECEPTORS_SET_PARAMETERS(pchMyReceptorSectionName, nReceptors, xn)
{
    if (!strcmp(pchMyReceptorSectionName, "R"))
        return prssG = new RasterSpikeSource;
    else if (!strcmp(pchMyReceptorSectionName, "Target"))
        return new LabelSpikeSource;
    else {
        cout << "objects_ArNI -- unknown section name: " << pchMyReceptorSectionName << endl;
        exit(-1);
    }
}
