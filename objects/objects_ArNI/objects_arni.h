#ifndef OBJECTS_ARNI_H
#define OBJECTS_ARNI_H

#define FOR_LINUX

#include <fcntl.h>
#include <unistd.h>

#include <sg/sg.h>
#include <NetworkConfigurator.h>

#include "objects_arni_int.h"

const int RecordPresentationPeriod = 15;
const int PicturePresentationTime = 10;
const int Width = 31;
const int Height = 31;
const double dmaxFrequency = 1.;
const unsigned ShuffleSeed = 0;
const bool bShuffle = false;
const int Block = 1;

class DYNAMIC_LIBRARY_EXPORTED_CLASS RasterSpikeSource: public IReceptors
{
    VECTOR<unsigned char>   vuc_Raster;
    VECTOR<double>          vd_State;
    int                     FrameTactCounter = 0;
    double                  dStateIncrementFactor;
    void GetSpikesfromImage(unsigned *pfl);
public:
    RasterSpikeSource(bool bLearning, int nReceptors);
    virtual bool bGenerateSignals(unsigned *pfl, int bitoffset) override;
    virtual void Randomize(void) override {}
    virtual void SaveStatus(Serializer &ser) const override {}
    virtual ~RasterSpikeSource(){}
    void LoadStatus(Serializer &ser);
};

class DYNAMIC_LIBRARY_EXPORTED_CLASS LabelSpikeSource: public IReceptors
{
    bool bGenerateSignalsI(unsigned *pfl);
    int GetPrediction(const VECTOR<PAIR<int, float> > &vpr_Votes) const;
    VECTOR<VECTOR<PAIR<int, float> > > vvpr_PredictedStates;
    bool                               bBeSilent;
public:
    LabelSpikeSource(bool bLearning);
    virtual bool bGenerateSignals(unsigned *pfl, int bitoffset) override
    {
        if (bitoffset)
            throw std::runtime_error("LabelSpikeSource -- multiple targets are not allowed");
        return bGenerateSignalsI(pfl);
    }
    void ObtainOutputSpikes(const VECTOR<int> &v_Firing);
    int         CurrentClass;
    size_t      tact = 0;
    VECTOR<int> vn_PredictionVotes;   // Its size is N target classes X N networks
};

#endif // OBJECTS_ARNI_H
