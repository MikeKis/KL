#ifndef OBJECTS_ARNI_H
#define OBJECTS_ARNI_H

#define FOR_LINUX

#include <boost/asio.hpp>
#include <boost/asio/posix/stream_descriptor.hpp>
#include <fcntl.h>
#include <unistd.h>

#include <sg/sg.h>
#include <NetworkConfigurator.h>

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
    VECTOR<unsigned char>                 vuc_Raster;
    VECTOR<double>                        vd_State;
    int                                   FrameTactCounter = 0;
    double                                dStateIncrementFactor;
    void GetSpikesfromImage(unsigned *pfl);
public:
    RasterSpikeSource();
    virtual bool bGenerateSignals(unsigned *pfl, int bitoffset) override;
    virtual void Randomize(void) override {}
    virtual void SaveStatus(Serializer &ser) const override {}
    void LoadStatus(Serializer &ser);
};

class DYNAMIC_LIBRARY_EXPORTED_CLASS LabelSpikeSource: public IReceptors
{
    bool bGenerateSignalsI(unsigned *pfl);
    int GetPrediction(const VECTOR<PAIR<int, float> > &vpr_Votes) const;
    boost::asio::io_context            io_ctx;
    int                                fd;
    VECTOR<VECTOR<PAIR<int, float> > > vvpr_PredictedStates;
public:
    LabelSpikeSource();
    virtual bool bGenerateSignals(unsigned *pfl, int bitoffset) override
    {
        if (bitoffset)
            throw std::runtime_error("LabelSpikeSource -- multiple targets are not allowed");
        return bGenerateSignalsI(pfl);
    }
    virtual ~LabelSpikeSource(){stream.close();}
    void ObtainOutputSpikes(const VECTOR<int> &v_Firing);
    int                                   CurrentClass;
    size_t                                tact = 0;
    VECTOR<int>                           vn_PredictionVotes;   // Its size is N target classes X N networks
    boost::asio::posix::stream_descriptor stream;
};

#endif // OBJECTS_ARNI_H
