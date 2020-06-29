#ifndef MyAnalysis_MyxAODAnalysis_H
#define MyAnalysis_MyxAODAnalysis_H

#include <AsgTools/ToolHandle.h>
#include <AnaAlgorithm/AnaAlgorithm.h>
#include <TH1.h>
#include <TTree.h>
#include <vector>

class MyxAODAnalysis : public EL::AnaAlgorithm
{
public:
  // this is a standard algorithm constructor
   MyxAODAnalysis (const std::string& name, ISvcLocator* pSvcLocator);
  
  // these are the functions inherited from Algorithm
  virtual StatusCode initialize () override;
  virtual StatusCode execute () override;
  virtual StatusCode finalize () override;

private:
  // Configuration, and any other types of variables go here.
  //float m_cutValue;
  //TTree *m_myTree;
  //TH1 *m_myHist;
  ~MyxAODAnalysis () override;

  unsigned int m_runNumber = 0; ///< Run number
  unsigned long long m_eventNumber = 0; ///< Event number
  unsigned int pvIndex = 0; //Index of the primary vertex
  
  std::vector<int> *m_jetCount = nullptr;
  
  std::vector<float> *m_jetMass = nullptr;
  std::vector<float> *m_jetEta = nullptr;
  std::vector<float> *m_jetPhi = nullptr;
  std::vector<float> *m_jetPt = nullptr;
  std::vector<float> *m_jetE = nullptr;

  std::vector<int> *m_jetNumTrkPt500 = nullptr;
  std::vector<int> *m_jetNumTrkPt1000 = nullptr;
  std::vector<float> *m_jetSumTrkPt500 = nullptr;
  std::vector<float> *m_jetSumTrkPt1000 = nullptr;
  std::vector<float> *m_jetTrackWidthPt500 = nullptr;
  std::vector<float> *m_jetTrackWidthPt1000 = nullptr;
  std::vector<float> *m_jetEMFrac = nullptr;
  std::vector<float> *m_jetHECFrac = nullptr;

  std::vector<float> *partE = nullptr;
  std::vector<float> *partPt = nullptr;
  std::vector<float> *partEta = nullptr;
  std::vector<float> *partPhi = nullptr;
  std::vector<float> *partMass = nullptr;
  std::vector<float> *partDeltaR = nullptr;
  std::vector<int> *partJetCount = nullptr;
  std::vector<int> *partRunNumber = nullptr;
  std::vector<int> *partEventNumber = nullptr;

  //bool c_isMC;
};

#endif
