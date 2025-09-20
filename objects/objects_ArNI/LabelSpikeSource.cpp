#include "objects_arni.h"

extern RasterSpikeSource *prssG;

int CurrentClass = 0;

bool LabelSpikeSource::bGenerateSignalsI(unsigned *pfl) const
{
    if (prssG->tact % RecordPresentationPeriod + 1 != PicturePresentationTime) {
        *reinterpret_cast<UNS64 *>(pfl) = 0;
        return true;
    }
    if (CurrentClass >= GetNReceptors())
        throw std::runtime_error("invalid class");
    *reinterpret_cast<UNS64 *>(pfl) |= 1ULL << CurrentClass;
    ++prssG->tact;
    return true;
}
