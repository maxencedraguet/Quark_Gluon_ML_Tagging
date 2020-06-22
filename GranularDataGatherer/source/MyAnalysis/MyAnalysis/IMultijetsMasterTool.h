#ifndef ATHNTUPFROMXAOD_IMULTIJETSMASTERTOOL_H
#define ATHNTUPFROMXAOD_IMULTIJETSMASTERTOOL_H 1

#include "AsgTools/IAsgTool.h"
#include "MyAnalysis/MJEnums.h"

#include "xAODBase/IParticle.h"

#include <TH1.h>

class IMultijetsMasterTool : public virtual asg::IAsgTool {
  public:
    ASG_TOOL_INTERFACE( IMultijetsMasterTool ) //declares the interface to athena

    virtual StatusCode readConfig() = 0;

    // Getters for the configurations
    virtual OutputSetting getOutputMode() const = 0;
    virtual OutputSetting getElectronMode() const = 0;
    virtual OutputSetting getMuonMode() const = 0;
    virtual OutputSetting getJetMode() const = 0;
    virtual OutputSetting getPhotonMode() const = 0;
    virtual bool getDoOutputPhotonCut() const = 0;
    virtual OutputSetting getTauMode() const = 0;
    virtual OutputSetting getTrackMode() const = 0;
    virtual OutputSetting getEtMissMode() const = 0;
    virtual OutputSetting getFatJetMode() const = 0;
    virtual bool getDoPhotonOR() const = 0;
    virtual bool getDoTauOR() const = 0;
    virtual bool getUsePhotonMET() const = 0;
    virtual bool getUseTauMET() const = 0;
    virtual bool getDoPRW() const = 0;
    virtual bool getIsSherpa() const = 0;
    virtual bool getIsTopTheoryWeight() const = 0;
    virtual bool getIsSingleTopTheoryWeight() const = 0;
    virtual bool getIsMC() const = 0;
    virtual bool isEmptyJob() const = 0;
    virtual bool getDoTileCorr() const = 0;
    virtual float getReclusteredJetMinPt() const = 0;
    virtual float getReclusteredJetMaxEta() const = 0;

    // Bookkeeping functions
    /***
     * Tells the tool whether or not the event is passed
     * If !isPassed increments the NEvents with no PV bin
     * If isPassed increments the Nevents executed bin
     */
    virtual void isEventPassed(bool isPassed) = 0;

    /***
     * Register a cut in the weighted/unweighted cutflows
     * Returns the bin number of the cut
     */
    virtual int registerCut(std::string cutName) = 0;

    /// Increment a bin in the cutflow (increments both the weighted and unweighted cutflows
    virtual void incrementCut(int cutBin) = 0;

    virtual TH1* getNumberOfEventsHist() = 0;

    virtual bool isFirstEvent() const = 0;

    // misc functions
    virtual void newFunctionWelcome(std::string functionName) = 0;

};

#endif //> !ATHNTUPFROMXAOD_IMULTIJETSMASTERTOOL_H
