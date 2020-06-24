#include <AsgTools/MessageCheck.h>
#include <MyAnalysis/MyxAODAnalysis.h>
#include <xAODEventInfo/EventInfo.h>
#include <xAODJet/JetContainer.h>

//#include "MyAnalysis/MJEnums.h"
//#include "MyAnalysis/IMultijetsMasterTool.h"
MyxAODAnalysis :: MyxAODAnalysis (const std::string& name,
                                  ISvcLocator *pSvcLocator)
    : EL::AnaAlgorithm (name, pSvcLocator)//, m_masterTool ("IMultijetsMasterTool")
{ 
  //declareProperty( "MasterTool", m_masterTool );
  // base variable initialisze
}

StatusCode MyxAODAnalysis :: initialize ()
{
  // to be done at beginning of worker node
  //ANA_CHECK (book (TH1F ("h_jetPt", "h_jetPt", 100, 0, 500))); // jet pt [GeV]
  //ANA_CHECK (book (TH1F ("h_deltaR", "h_deltaR", 100, 0, 1))); // deltaR between jet and constituent particle

  //m_masterTool->newFunctionWelcome("MultijetsSystematicNTUPMaker::initialize");
  //ATH_CHECK( m_masterTool.retrieve() );
  //c_isMC = m_masterTool->getIsMC();
  //ANA_MSG_INFO ("is MC ? "<<c_isMC);
  ANA_CHECK (book (TTree ("analysis", "My analysis ntuple")));
  TTree* mytree = tree ("analysis");
  mytree->Branch ("RunNumber", &m_runNumber);
  mytree->Branch ("EventNumber", &m_eventNumber);
 
  m_jetCount = new std::vector<int>();
  mytree->Branch ("jetNumber", &m_jetCount);
  //Basic Jet info
  m_jetMass = new std::vector<float>();
  mytree->Branch ("jetMass", &m_jetMass);
  m_jetEta = new std::vector<float>();
  mytree->Branch ("jetEta", &m_jetEta);
  m_jetPhi = new std::vector<float>();
  mytree->Branch ("jetPhi", &m_jetPhi);
  m_jetPt = new std::vector<float>();
  mytree->Branch ("jetPt", &m_jetPt);
  m_jetE = new std::vector<float>();
  mytree->Branch ("jetE", &m_jetE);
  //Jet momentum
  m_jetNumTrkPt500= new std::vector<int>();
  mytree->Branch ("jetNumTrkPt500", &m_jetNumTrkPt500);
  m_jetNumTrkPt1000= new std::vector<int>(); 
  mytree->Branch ("jetNumTrkPt1000", &m_jetNumTrkPt1000);
  m_jetSumTrkPt500 = new std::vector<float>();
  mytree->Branch ("jetSumTrkPt500", &m_jetSumTrkPt500);
  m_jetSumTrkPt1000 = new std::vector<float>();
  mytree->Branch ("jetSumTrkPt1000", &m_jetSumTrkPt1000);
  m_jetTrackWidthPt500 = new std::vector<float>();
  mytree->Branch ("jetTrackWidthPt500", &m_jetTrackWidthPt500);
  m_jetTrackWidthPt1000 = new std::vector<float>();
  mytree->Branch ("jetTrackWidthPt1000", &m_jetTrackWidthPt1000);
  m_jetEMFrac = new std::vector<float>();
  mytree->Branch ("jetEMFrac", &m_jetEMFrac);
  m_jetHECFrac = new std::vector<float>();
  mytree->Branch ("jetHECFrac", &m_jetHECFrac);
  //m_jetChFrac = new std::vector<float>();
  //mytree->Branch ("jetChFrac", &m_jetChFrac);
  //Not clear where to get the last one

  //Constituents entries
  partE = new std::vector<float>();
  mytree->Branch ("constituentE", &partE);
  partPt = new std::vector<float>();
  mytree->Branch ("constituentPt", &partPt);
  partEta = new std::vector<float>();
  mytree->Branch ("constituentEta", &partEta);
  partPhi = new std::vector<float>();
  mytree->Branch ("constituentPhi", &partPhi);
  partMass = new std::vector<float>();
  mytree->Branch ("constituentMass", &partMass);
  partDeltaR = new std::vector<float>();
  mytree->Branch ("constituentDeltaRtoJet", &partDeltaR);
  partJetCount = new std::vector<int>();
  mytree->Branch ("constituentJet", &partJetCount);
  partRunNumber  = new std::vector<int>();
  mytree->Branch ("constituentRunNumber", &partRunNumber);
  partEventNumber = new std::vector<int>();
  mytree->Branch ("constituentEventNumber", &partEventNumber);
  return StatusCode::SUCCESS;
}

StatusCode MyxAODAnalysis :: execute ()
{  
  //static SG::AuxElement::ConstAccessor<unsigned int>       cacc_pvIndex("pvIndex");
  //static SG::AuxElement::ConstAccessor<char>               cacc_bjet("bjet");
  /* to be done on every events  
  const xAOD::JetContainer* jets = nullptr;
  ANA_CHECK (evtStore()->retrieve (jets, "AntiKt4TruthJets"));
  ANA_MSG_INFO ("execute(): number of jets = " << jets->size());
  for (const xAOD::Jet* jet : *jets) {
    ANA_MSG_INFO ("execute(): jet pt = " << (jet->pt() * 0.001) << " GeV");
  }
  */
  const xAOD::EventInfo* ei = nullptr;
  ANA_CHECK (evtStore()->retrieve (ei, "EventInfo"));
  const xAOD::EventInfo* evtInfo2(0);
  ATH_CHECK(evtStore()->retrieve(evtInfo2, "EventInfo" ) );
  m_runNumber = ei->runNumber ();
  m_eventNumber = ei->eventNumber ();
  const xAOD::JetContainer* jets2 = nullptr;
  ANA_CHECK (evtStore()->retrieve (jets2, "AntiKt4EMTopoJets"));
 
  m_jetMass->clear();
  m_jetEta->clear();
  m_jetPhi->clear();
  m_jetPt->clear();
  m_jetE->clear();
 
  m_jetNumTrkPt500->clear();
  m_jetNumTrkPt1000->clear();
  m_jetSumTrkPt500->clear();
  m_jetSumTrkPt1000->clear();
  m_jetTrackWidthPt500->clear();
  m_jetTrackWidthPt1000->clear();
  m_jetEMFrac->clear();
  m_jetHECFrac->clear();
  //m_jetChFrac->clear();
  
  int counter_jet = 0;
  int counter_part= 0; 

  partE->clear();
  partPt->clear();
  partEta->clear();
  partPhi->clear();
  partMass->clear();
  partJetCount->clear();
  partDeltaR->clear();
  partRunNumber->clear();
  partEventNumber->clear();
 //unsigned int pvIndex = cacc_pvIndex(*evtInfo2); 
  //partConstituentCount->clear();
  //partRunNumber->clear();
  //partEventNumber->clear();
 for (const xAOD::Jet* jet : *jets2) {
    counter_jet ++;

    //ANA_MSG_INFO ("execute(): B jet "<< cacc_bjet(*jet) );
    m_jetEta->push_back (jet->eta ());
    m_jetPhi->push_back (jet->phi ());
    m_jetPt-> push_back (jet->pt () * 0.001);
    m_jetE->  push_back (jet->e ()* 0.001);
    m_jetMass->push_back(jet->m());
    m_jetCount->push_back (counter_jet);
    //std::vector<const int*> pobjs  
    std::vector<int> pobjs = jet->getAttribute<std::vector<int>>("NumTrkPt500");
    std::vector<float> pobjs2=jet->getAttribute<std::vector<float>>("SumPtTrkPt500");
    //m_jetNumTrkPt500->push_back(jet->getAssociatedObjects<int> ( "NumTrkPt500") );
    /*
    m_jetNumTrkPt1000->push_back(jet->getAttribute<int>( "NumTrkPt1000")); 

    m_jetSumTrkPt500->push_back(jet->getAttribute<float>("SumPtTrkPt500"));
    m_jetNumTrkPt1000->push_back(jet->getAttribute<float>("SumPtTrkPt1000"));

    m_jetTrackWidthPt500->push_back(jet->getAttribute<float>("TrackWidthPt500"));
    m_jetTrackWidthPt1000->push_back(jet->getAttribute<float>("TrackWidthPt1000"));
    */
    m_jetEMFrac-> push_back(jet->getAttribute<float>("EMFrac"));
    //m_jetHECFrac->push_back(jet->getAttribute<float>("HECFrac")); //problem is that it is non existent for AntiKt4EMTopoJets
    ANA_MSG_INFO ("execute(): jet m_jetEMFrac "<< jet->getAttribute<float>("EMFrac"));
    //ANA_MSG_INFO ("execute(): jet m_jetNumTrkPt500 = " << (jet->getAttribute<float>("SumPtTrkPt500")));
    //ANA_MSG_INFO ("execute(): jet m_jetNumTrkPt500 = " << (jet->getAttribute<int>( "NumTrkPt500")));
    //ANA_MSG_INFO ("execute(): jet m_jetNumTrkPt500 size= " <<pobjs.size());
    //ANA_MSG_INFO ("execute(): jet m_jetNumTrkPt500 AT PV INDEX = " << jet->getAttribute<std::vector<int>>(xAOD::JetAttribute::NumTrkPt500).at(pvIndex) );
    for (unsigned i=0; i<pobjs.size(); ++i)
    	ANA_MSG_INFO ("execute(): m_jetNumTrkPt500 " << pobjs[i]);
   
    ANA_MSG_INFO ("execute(): jet m_jetSumTrkPt500 size= " <<pobjs2.size());
    for (unsigned i=0; i<pobjs2.size(); ++i)
        ANA_MSG_INFO ("execute(): m_jetSumTrkPt500 " << pobjs2[i]);
    //ANA_MSG_INFO ("execute(): jet eta= " << jet->eta());
    //ANA_MSG_INFO ("execute(): jet phi= " << jet->phi());
     
    //qg nTrack tagger
    /*
    ATH_MSG_DEBUG( "Entering qg tagger" );
    Root::TAccept qg_taccept = m_qgTagger->tag( *jet, nullptr );
    isnTrkQuark.push_back(qg_taccept.getCutResult(4));
    */

    counter_part = 0;
    const xAOD::JetConstituentVector cons = jet->getConstituents();
    for (auto cluster_itr : cons){
        counter_part ++;
        ANA_MSG_INFO ("execute(): processing event: " << m_runNumber << " eventnumber: " << m_eventNumber << " for jet number " << counter_jet <<" in particle number: "<<counter_part);
        partE->push_back(cluster_itr->e() * 0.001);
    	partPt->push_back(cluster_itr->pt() * 0.001);
    	partEta->push_back(cluster_itr->eta());
    	partPhi->push_back(cluster_itr->phi());
        partMass->push_back(cluster_itr->m());
        partJetCount->push_back(counter_jet);
        partRunNumber->push_back(m_runNumber);
        partEventNumber->push_back(m_eventNumber);	

	//Compute delta R between consitutent and Jet
        Double_t deta = cluster_itr->eta() - jet->eta ();
        Double_t dphi = TVector2::Phi_mpi_pi(cluster_itr->phi() - jet->phi());
        double_t dR   = TMath::Sqrt( deta*deta+dphi*dphi );
        //hist ("h_deltaR")->Fill (dR);
        partDeltaR->push_back(dR);
  	/*
        ANA_MSG_INFO ("execute(): constitutent Run " <<m_runNumber<< " and eventnumber "<<m_eventNumber);
    	ANA_MSG_INFO ("execute(): constitutent jet pt = " << (cluster_itr->pt() * 0.001) << " GeV");
    	ANA_MSG_INFO ("execute(): constitutent jet eta= " << cluster_itr->eta());
    	ANA_MSG_INFO ("execute(): constitutent jet phi= " << cluster_itr->phi());
	ANA_MSG_INFO ("execute(): constitutent to jet dR= " << dR);
	*/
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
  delete m_jetCount;
  delete m_jetMass;
  delete m_jetEta;
  delete m_jetPhi;
  delete m_jetPt;
  delete m_jetE;
  delete m_jetNumTrkPt500;
  delete m_jetNumTrkPt1000;
  delete m_jetSumTrkPt500;
  delete m_jetSumTrkPt1000;
  delete m_jetTrackWidthPt500;
  delete m_jetTrackWidthPt1000;
  delete m_jetEMFrac;
  delete m_jetHECFrac;

  delete partE;
  delete partPt;
  delete partPhi;
  delete partEta;
  delete partMass;
  delete partJetCount;
  delete partDeltaR;
  delete partRunNumber; 
  delete partEventNumber;
}
