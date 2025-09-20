#ifndef OBJECTS_ARNI_H
#define OBJECTS_ARNI_H

#define FOR_LINUX

#include <boost/asio.hpp>
#include <boost/asio/posix/stream_descriptor.hpp>
#include <fcntl.h>
#include <unistd.h>

#include <sg/sg.h>

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
    boost::asio::io_context               io_ctx;
    int                                   fd;
    boost::asio::posix::stream_descriptor stream;
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
    virtual ~RasterSpikeSource(){stream.close();}
    size_t tact = 0;
};

class DYNAMIC_LIBRARY_EXPORTED_CLASS LabelSpikeSource: public IReceptors
{
    bool bGenerateSignalsI(unsigned *pfl) const;
public:
    virtual bool bGenerateSignals(unsigned *pfl, int bitoffset) override
    {
        if (bitoffset)
            throw std::runtime_error("LabelSpikeSource -- multiple targets are not allowed");
        return bGenerateSignalsI(pfl);
    }
};

#endif // OBJECTS_ARNI_H
