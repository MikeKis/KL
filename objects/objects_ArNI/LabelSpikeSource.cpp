#include <iostream>

#include "objects_arni.h"
#include "objects_arni_int.h"

using namespace std;
namespace asio = boost::asio;

LabelSpikeSource::LabelSpikeSource(): stream(io_ctx)
{
    fd = open(pchArNI_fifo_path, O_RDWR);
    if (fd == -1) {
        cout << "Error opening FIFO for reading: " << strerror(errno) << std::endl;
        exit(-40000);
    }
    stream.assign(fd);
}

bool LabelSpikeSource::bGenerateSignalsI(unsigned *pfl)
{
    if (tact % RecordPresentationPeriod + 1 != PicturePresentationTime) {
        *reinterpret_cast<UNS64 *>(pfl) = 0;
        return true;
    }
    if (CurrentClass >= GetNReceptors())
        throw std::runtime_error("invalid class");
    *reinterpret_cast<UNS64 *>(pfl) |= 1ULL << CurrentClass;
    return true;
}

void getrelvotes(vector<int>::const_iterator votesbeg, int nVariants, vector<float> &vr_RelativeVotes)
{
    vr_RelativeVotes.clear();
    int ntotvotes = accumulate(votesbeg, votesbeg + nVariants, 0);
    if (!ntotvotes)
        return;
    vr_RelativeVotes.resize(nVariants);
    for (int i = 0; i < nVariants; ++i)
        vr_RelativeVotes[i] = *votesbeg++ / (float)ntotvotes;
}

void LabelSpikeSource::ObtainOutputSpikes(const VECTOR<int> &v_Firing)
{
    for (auto i: v_Firing)
        ++vn_PredictionVotes[i];
    if (!((tact + 1) % RecordPresentationPeriod)) {   // tact 14
        static int indran = 0;
        vector<float> vr_perNetworkPredictionVotes(GetNReceptors(), 0.F);
        auto l = vn_PredictionVotes.begin();
        vector<vector<float> > vvr_(GetNReceptors());   // relative votes for each class (only for networks with at least 1 vote)
        while (l != vn_PredictionVotes.end()) {
            vector<float> vr_RelativeVotes;
            getrelvotes(l, GetNReceptors(), vr_RelativeVotes);
            if (vr_RelativeVotes.size())
                FORI(GetNReceptors()) {
                    vvr_[_i].push_back(vr_RelativeVotes[_i]);
                    vr_perNetworkPredictionVotes[_i] += vr_RelativeVotes[_i];   // Может быть суммировать не относительные голоса, а абсолютные?
                }
            l += GetNReceptors();
        }
        vector<pair<int, float> > vpr_;
        FORI(GetNReceptors())
            if (vr_perNetworkPredictionVotes[_i])
                vpr_.push_back(pair<int, float>(_i, vr_perNetworkPredictionVotes[_i]));
        vvpr_PredictedStates.push_back(vpr_);
        int pred = GetPrediction(vpr_);   // in fact it is not used in training mode
        asio::write(stream, asio::buffer(&pred, sizeof(pred)));
        fill(vn_PredictionVotes.begin(), vn_PredictionVotes.end(), 0);
    }
    ++tact;
}

int LabelSpikeSource::GetPrediction(const vector<pair<int, float> > &vpr_Votes) const
{
    vector<float> vr_perNetworkPredictionVotes(GetNReceptors(), 0.F);
    for (auto i: vpr_Votes)
        vr_perNetworkPredictionVotes[i.first] = i.second;
    static vector<float> vr_;
    if (vr_.empty())
        vr_.resize(GetNReceptors(), 0.F);
    FORI(GetNReceptors())
        vr_[_i] += vr_perNetworkPredictionVotes[_i];
    float rbest = 0.F;
    int pred;
    static int scount = 0;
    int indran = (int)(scount++ % vr_.size());
    FORI(vr_.size()) {
        if (vr_[indran] > rbest) {
            rbest = vr_[indran];
            pred = indran;
        }
        if (++indran == vr_.size())
            indran = 0;
    }
    fill(vr_.begin(), vr_.end(), 0.F);
    return pred;
}
