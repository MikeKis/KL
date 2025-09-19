#ifndef OBJECTS_ARNI_H
#define OBJECTS_ARNI_H

#define FOR_LINUX

#include <sg/sg.h>

class DYNAMIC_LIBRARY_EXPORTED_CLASS RasterSpikeSource: public IReceptors
{
    size_t tact = 0;




    VECTOR<VECTOR<unsigned> > vvfl_InputSignal;
    VECTOR<unsigned>          vfl_InputSignal;
    VECTOR<unsigned char>     vuc_Raster;
    enum mode
    {
        normal,
        repetitive,
        endless,
        invalid
    }                         mod;
    VECTOR<STRING>            vstr_meanings;
    size_t                    HistoryLength;
    enum type
    {
        plain_record_binary,
        plain_record_text,
        text_values,
        picture,
        no_input
    }                         typ = plain_record_text;
    int						  ntactperPicture = 0;
    int                       FrameTactCounter = 0;
    VECTOR<double>            vd_State;
    double                    dStateIncrementFactor;
    double                    dState = 0.;
    double                    dFrequencyMultiplier = 1.;
    Rand                      rng;
    float                     rNoise = 0.F;
    int                       Period = 0;

    void GetSpikesfromImage(int nPixels, VECTOR<unsigned> &vfl_InpSig);
    void DropSpikes(int nPixels, VECTOR<unsigned> &vfl_InpSig);
    void AddNoise(int nPixels, VECTOR<unsigned> &vfl_InpSig);
    void GetSpikesfromTextValuesFile(
        STRING strSource,
        int RecordPresentationTime,
        int RecordPresentationPeriod,
        int nReceptiveFields,
        unsigned ShuffleSeed,
        int BlockSize,
        PAIR<int, int> pindind_GaussianCalibtrationRecords,   // It is assumed that this is the record range used for learning - so that only records in this
        // range are shuffled.
        const VECTOR<pugi::xml_node> &vxn_InputVariables
        );

public:
    RasterSpikeSource() {};
    virtual bool bGenerateSignals(unsigned *pfl, int bitoffset) override;



    virtual void Randomize(void) override;
    virtual void SaveStatus(Serializer &ser) const override;
    virtual void GetMeanings(VECTOR<STRING> &vstr_Meanings) const override
    {
        if (vstr_meanings.empty())
            IReceptors::GetMeanings(vstr_Meanings);
        else vstr_Meanings = vstr_meanings;
    }
    void LoadStatus(Serializer &ser);
    virtual ~FileSpikeSource() = default;
};

class DYNAMIC_LIBRARY_EXPORTED_CLASS LabelSpikeSource: public IReceptors
{

};


const int RecordPresentationPeriod = 15;
const int PicturePresentationTime = 10;
const int Width = 31;
const int Height = 31;
const double dmaxFrequency = 1.;
const unsigned ShuffleSeed = 0;
const bool bShuffle = false;
const int Block = 1;

#endif // OBJECTS_ARNI_H
