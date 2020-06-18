#include <AsgTools/MessageCheck.h>
#include <MyAnalysis/MyxAODAnalysis.h>
#include <xAODEventInfo/EventInfo.h>
#include <xAODJet/JetContainer.h>

MyxAODAnalysis :: MyxAODAnalysis (const std::string& name,
                                  ISvcLocator *pSvcLocator)
    : EL::AnaAlgorithm (name, pSvcLocator)
{
  // base variable initialisze
}

StatusCode MyxAODAnalysis :: initialize ()
{
  // to be done at beginning of worker node
  ANA_CHECK (book (TH1F ("h_jetPt", "h_jetPt", 100, 0, 500))); // jet pt [GeV]
  
  ANA_CHECK (book (TTree ("analysis", "My analysis ntuple")));
  TTree* mytree = tree ("analysis");
  mytree->Branch ("RunNumber", &m_runNumber);
  mytree->Branch ("EventNumber", &m_eventNumber);
  m_jetEta = new std::vector<float>();
  mytree->Branch ("JetEta", &m_jetEta);
  m_jetPhi = new std::vector<float>();
  mytree->Branch ("JetPhi", &m_jetPhi);
  m_jetPt = new std::vector<float>();
  mytree->Branch ("JetPt", &m_jetPt);
  m_jetE = new std::vector<float>();
  mytree->Branch ("JetE", &m_jetE);
  partPt = new std::vector<float>();
  mytree->Branch ("constituentPt", &partPt);
  partEta = new std::vector<float>();
  mytree->Branch ("constituentPt", &partEta);
  partPhi = new std::vector<float>();
  mytree->Branch ("constituentPt", &partPhi);
  return StatusCode::SUCCESS;
}

StatusCode MyxAODAnalysis :: execute ()
{
  // to be done on every events  
  const xAOD::JetContainer* jets = nullptr;
  ANA_CHECK (evtStore()->retrieve (jets, "AntiKt4TruthJets"));
  ANA_MSG_INFO ("execute(): number of jets = " << jets->size());
  for (const xAOD::Jet* jet : *jets) {
    ANA_MSG_INFO ("execute(): jet pt = " << (jet->pt() * 0.001) << " GeV");
  }
  const xAOD::EventInfo* ei = nullptr;
  ANA_CHECK (evtStore()->retrieve (ei, "EventInfo"));
  m_runNumber = ei->runNumber ();
  m_eventNumber = ei->eventNumber ();
  const xAOD::JetContainer* jets2 = nullptr;
  ANA_CHECK (evtStore()->retrieve (jets2, "AntiKt4EMTopoJets"));
  m_jetEta->clear();
  m_jetPhi->clear();
  m_jetPt->clear();
  m_jetE->clear();
  int counter_jet = 0;
  int counter_part= 0; 
  //const xAOD::JetConstituentVector cons;
  partPt->clear();
  partEta->clear();
  partPhi->clear();
 for (const xAOD::Jet* jet : *jets2) {
    counter_jet ++;
    m_jetEta->push_back (jet->eta ());
    m_jetPhi->push_back (jet->phi ());
    m_jetPt-> push_back (jet->pt ());
    m_jetE->  push_back (jet->e ());
    counter_part = 0;
    const xAOD::JetConstituentVector cons = jet->getConstituents();
    for (auto cluster_itr : cons){
        ANA_MSG_INFO ("execute(): processing event: " << m_runNumber << " eventnumber: " << m_eventNumber << "for jet number " << counter_jet <<" in particle number: "<<counter_part);
        counter_part ++;
    	partPt->push_back(cluster_itr->pt());
    	partEta->push_back(cluster_itr->eta());
    	partPhi->push_back(cluster_itr->phi());
    	ANA_MSG_INFO ("execute(): constitutent jet pt = " << (cluster_itr->pt() * 0.001) << " GeV");
    	ANA_MSG_INFO ("execute(): constitutent jet eta= " << cluster_itr->eta());
    	ANA_MSG_INFO ("execute(): constitutent jet phi= " << cluster_itr->phi());
    }   
  }
  tree ("analysis")->Fill ();
  
  return StatusCode::SUCCESS;
}

StatusCode MyxAODAnalysis :: finalize ()
{
  // to be done after the last event
  return StatusCode::SUCCESS;
}

MyxAODAnalysis::~MyxAODAnalysis(){
  delete m_jetEta;
  delete m_jetPhi;
  delete m_jetPt;
  delete m_jetE;
  delete partPt;
  delete partPhi;
  delete partEta;
}
