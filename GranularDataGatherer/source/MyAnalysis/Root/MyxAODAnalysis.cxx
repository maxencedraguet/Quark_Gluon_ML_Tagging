#include <AsgTools/MessageCheck.h>
#include <MyAnalysis/MyxAODAnalysis.h>
#include <xAODEventInfo/EventInfo.h>
#include <xAODJet/JetContainer.h>
#include <xAODCore/ShallowCopy.h>

MyxAODAnalysis :: MyxAODAnalysis (const std::string& name,
                                  ISvcLocator *pSvcLocator)
    : EL::AnaAlgorithm (name, pSvcLocator),
    m_grl ("GoodRunsListSelectionTool/grl", this),
    m_SUSYTools ("ST::ISUSYObjDef_xAODTool/SUSYTools", this)

{ 
  // base variable initialisze
  declareProperty ("grlTool", m_grl, "the GRL tool");
  declareProperty ("SUSYTools", m_SUSYTools);
}

StatusCode MyxAODAnalysis :: initialize ()
{
  /////////////////////////////////////////
  //
  // Tool Initialistaion.
  //
  /////////////////////////////////////////

  ANA_CHECK (m_grl.retrieve());
  ANA_CHECK (m_SUSYTools.retrieve());

  /////////////////////////////////////////
  //
  // Book the tree and setup the branches.
  //
  /////////////////////////////////////////

  ANA_CHECK (book (TTree ("analysis", "My analysis ntuple")));
  TTree* mytree = tree ("analysis");
  mytree->Branch ("RunNumber", &m_runNumber);
  mytree->Branch ("EventNumber", &m_eventNumber);
 
  //Jets
  m_jetCount = new std::vector<int>();
  mytree->Branch ("jetNumber", &m_jetCount);
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
  m_jetWidth = new std::vector<float>();
  mytree->Branch ("jetWidth", &m_jetWidth);
  m_jetNumberConstituent = new std::vector<int>();
  mytree->Branch ("jetNumberConstituent", &m_jetNumberConstituent);

  //Jet quality cuts.
  isNotPVJet = new std::vector<int>();
  mytree->Branch ("isNotPVJet", &isNotPVJet);
  isBadJet = new std::vector<char>();
  mytree->Branch ("isBadJet", &isBadJet);
  isBaselineJet = new std::vector<char>();
  mytree->Branch ("isBaselineJet", &isBaselineJet);
  isSignalJet = new std::vector<char>();
  mytree->Branch ("isSignalJet", &isSignalJet);
  isBJet = new std::vector<char>();
  mytree->Branch ("isBJet", &isBJet);
  passJvt = new std::vector<char>();
  mytree->Branch ("passJvt", &passJvt);
  passfJvt = new std::vector<char>();
  mytree->Branch ("passfJvt", &passfJvt);
  JvtScore = new std::vector<float>();
  mytree->Branch ("JvtScore", &JvtScore);
  fJvtScore = new std::vector<float>();
  mytree->Branch ("fJvtScore", &fJvtScore);
  btag_weight = new std::vector<double>();
  mytree->Branch ("btag_weight", &btag_weight);

  
  //Jet constituents
  m_jetNumTrkPt500 = new std::vector<int>();
  mytree->Branch ("jetNumTrkPt500", &m_jetNumTrkPt500);
  m_jetNumTrkPt1000 = new std::vector<int>(); 
  mytree->Branch ("jetNumTrkPt1000", &m_jetNumTrkPt1000);
  m_jetSumTrkPt500 = new std::vector<float>();
  mytree->Branch ("jetSumTrkPt500", &m_jetSumTrkPt500);
  m_jetSumTrkPt1000 = new std::vector<float>();
  mytree->Branch ("jetSumTrkPt1000", &m_jetSumTrkPt1000);
  m_jetTrackWidthPt500 = new std::vector<float>();
  mytree->Branch ("jetTrackWidthPt500", &m_jetTrackWidthPt500);
  m_jetTrackWidthPt1000 = new std::vector<std::vector<float>>();
  mytree->Branch ("jetTrackWidthPt1000", &m_jetTrackWidthPt1000);
    
  isTruthQuark= new std::vector<int>();
  mytree->Branch ("isTruthQuark", &isTruthQuark);
  partonID = new std::vector<int>();
  mytree->Branch ("partonID", &partonID);


  //Calorimeter variables
  m_jetEMFrac = new std::vector<float>();
  mytree->Branch ("jetEMFrac", &m_jetEMFrac);
  m_jetHECFrac = new std::vector<float>();
  mytree->Branch ("jetHECFrac", &m_jetHECFrac);
  m_jetChFrac = new std::vector<float>();
  mytree->Branch ("jetChFrac", &m_jetChFrac);

  //Constituents entries
  partE = new std::vector<float>();
  mytree->Branch ("constituentE", &partE);
  partPt = new std::vector<float>();
  mytree->Branch ("constituentPt", &partPt);
  partPx = new std::vector<float>();
  mytree->Branch ("constituentPx", &partPx);
  partPy = new std::vector<float>();
  mytree->Branch ("constituentPy", &partPy);
  partPz = new std::vector<float>();
  mytree->Branch ("constituentPz", &partPz);
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
  /////////////////////////////////////////
  //
  // Define all accessors.
  //
  /////////////////////////////////////////

  //SUSY Tools Accessors (these are available after 'GetJets()' method).
  static SG::AuxElement::ConstAccessor<char> cacc_bad("bad");
  static SG::AuxElement::ConstAccessor<char> cacc_baseline("baseline");
  static SG::AuxElement::ConstAccessor<char> cacc_signal("signal");
  static SG::AuxElement::ConstAccessor<char> cacc_bjet("bjet");
  static SG::AuxElement::ConstAccessor<char> cacc_passJvt("passJvt");
  static SG::AuxElement::ConstAccessor<char> cacc_passFJvt("passFJvt");
  static SG::AuxElement::ConstAccessor<float> cacc_jvt("Jvt");
  static SG::AuxElement::ConstAccessor<float> cacc_fjvt("fJvt");
  static SG::AuxElement::ConstAccessor<double> cacc_btag_weight("btag_weight");

  /////////////////////////////////////////
  //
  // Clear all vars (vectors first).
  //
  /////////////////////////////////////////

  //Jet vectors.
  m_jetMass->clear();
  m_jetEta->clear();
  m_jetPhi->clear();
  m_jetPt->clear();
  m_jetE->clear();
  m_jetWidth->clear();
  m_jetCount->clear();
  m_jetNumberConstituent->clear();
    
  //Jet quality and tagging.
  isNotPVJet->clear();
  isBJet->clear();
  isBaselineJet->clear();
  isSignalJet->clear();
  isBJet->clear();
  passJvt->clear();
  passfJvt->clear();
  JvtScore->clear();
  fJvtScore->clear();
  btag_weight->clear();
 
  //Jet track associated values.
  m_jetNumTrkPt500->clear();
  m_jetNumTrkPt1000->clear();
  m_jetSumTrkPt500->clear();
  m_jetSumTrkPt1000->clear();
  m_jetTrackWidthPt500->clear();
  m_jetTrackWidthPt1000->clear();
  partonID->clear();
  isTruthQuark->clear();
  
  //Jet calorimeter fractions.
  m_jetEMFrac->clear();
  m_jetHECFrac->clear();
  m_jetChFrac->clear();
   
  //Jet constituent values (EMTopo ONLY! Add PFlow EMTopo split.) 
  partE->clear();
  partPt->clear();
  partPx->clear();
  partPy->clear();
  partPz->clear();
  partEta->clear();
  partPhi->clear();
  partMass->clear();
  partJetCount->clear();
  partDeltaR->clear();
  partRunNumber->clear();
  partEventNumber->clear();

  //Jet and parton counters.
  int counter_jet = 0;
  int counter_part= 0;

  //vertexing
  pvIndex = 0;

  /////////////////////////////////////////
  //
  // Event Information.
  //
  /////////////////////////////////////////

  const xAOD::EventInfo* ei = nullptr;
  ANA_CHECK (evtStore()->retrieve (ei, "EventInfo"));

  //Is it MC?
  bool isMC = false;
  if (ei->eventType (xAOD::EventInfo::IS_SIMULATION)) {
    isMC = true;
  }
  //ANA_MSG_INFO ("is MC: " << isMC);

  //Retrieve event and run number.
  m_runNumber = ei->runNumber ();
  m_eventNumber = ei->eventNumber ();

  //Data needs to pass the GRL (Good Runs List) conditions and filter out 
  //errors in the various detector subsystems.
  if (!isMC) {
    if (!m_grl->passRunLB(*ei)) {
      ANA_MSG_INFO ("Event: " << m_runNumber << m_eventNumber << " failed GRL.");
      return StatusCode::SUCCESS;
    }       
    if ((ei->errorState(xAOD::EventInfo::Tile)==xAOD::EventInfo::Error)){
      ANA_MSG_INFO ("Event: " << m_runNumber << m_eventNumber << " failed Tile.");
      return StatusCode::SUCCESS;      
    }
    if ((ei->errorState(xAOD::EventInfo::LAr)==xAOD::EventInfo::Error)){
      ANA_MSG_INFO ("Event: " << m_runNumber << m_eventNumber << " failed LAr.");
      return StatusCode::SUCCESS;      
    }
    if ((ei->errorState(xAOD::EventInfo::SCT)==xAOD::EventInfo::Error)){
      ANA_MSG_INFO ("Event: " << m_runNumber << m_eventNumber << " failed SCT.");
      return StatusCode::SUCCESS;      
    }    
    if ((ei->isEventFlagBitSet(xAOD::EventInfo::Core, 18))){
      ANA_MSG_INFO ("Event: " << m_runNumber << m_eventNumber << " incomplete event.");
      return StatusCode::SUCCESS;      
    }    
  }

  //Get the primary vertex and its index.
  const xAOD::Vertex* priVtx = m_SUSYTools->GetPrimVtx();
  //Must have one primary vertex with more than two tracks or event is bad.
  if (!priVtx || priVtx->nTrackParticles() < 2){
    ANA_MSG_INFO ("Event: " << m_runNumber << m_eventNumber << " failed vertex selection.");
    return StatusCode::SUCCESS;
  }
  //Get the PV index (Maxence, usually we decorate here but there is no need
  //as we are using it in the same algorithm).
  pvIndex = priVtx->index();

  //Debug messages to check status of primary vertex
  ANA_MSG_DEBUG("Primary vertex: " << pvIndex);
  ANA_MSG_DEBUG("PV nTrack: " << priVtx->nTrackParticles());

  /////////////////////////////////////////
  //
  // Jets.
  //
  /////////////////////////////////////////

  //Get EMTopo jets container (add option for other jet types later).
  //Standard Method.
  const xAOD::JetContainer* jets = nullptr;
  ANA_CHECK(evtStore()->retrieve(jets, "AntiKt4EMTopoJets"));
  //Standard Method.

  //SUSY Tools method.
  xAOD::JetContainer* jets_nominal(0);
  xAOD::ShallowAuxContainer* jets_nominal_aux(0);

  //Now get the jets with the main and aux container to decorate the cuts.
  ANA_CHECK(m_SUSYTools->GetJets(jets_nominal, jets_nominal_aux, true));
  //SUSY Tools method.

  //Loop over the shallow copied jets.
  for (auto jet : *jets_nominal) {
    //Jet quality and tagging quantities.
    isBadJet->push_back(cacc_bad(*jet));
    isBaselineJet->push_back(cacc_baseline(*jet));
    isSignalJet->push_back(cacc_signal(*jet));
    isBJet->push_back(cacc_bjet(*jet));
    passJvt->push_back(cacc_passJvt(*jet));
    passfJvt->push_back(cacc_passFJvt(*jet));
    JvtScore->push_back(cacc_jvt(*jet));
    fJvtScore->push_back(cacc_fjvt(*jet));
    btag_weight->push_back(cacc_btag_weight(*jet));

    
    //Push back all the main jet vector quantities.
    m_jetEta->push_back(jet->eta());
    m_jetPhi->push_back(jet->phi());
    m_jetPt-> push_back(jet->pt());
    m_jetE->push_back(jet->e());
    m_jetWidth->push_back(jet->getAttribute<float>("Width"));
    m_jetMass->push_back(jet->m());
    m_jetCount->push_back (counter_jet); //To control constituent-jet matching.

    //Truth information.
    if (isMC) {
      int partonID_value = jet->getAttribute<int>("PartonTruthLabelID");
      if (partonID_value > 0 && partonID_value <= 6) {
          isTruthQuark->push_back(1);
      }
      if (partonID_value == 21) {
          isTruthQuark->push_back(0);
      }
      if (partonID_value == -1) {
          isTruthQuark->push_back(-1);
      }
      partonID->push_back(partonID_value);
    }

    //Calculate jet charge fraction.
    m_jetChFrac->push_back(jet->getAttribute<std::vector<float>>("SumPtTrkPt500").at(pvIndex) / jet->pt());

    //Fraction of jet energy from EM calo.
    m_jetEMFrac->push_back(jet->getAttribute<float>("EMFrac"));

    //Hadronic calorimeter fraction.
    //m_jetHECFrac->push_back(jet->getAttribute<float>("HECFrac"));
 
    //Push back all the jet substructure quantities.
    m_jetNumTrkPt500->push_back(jet->getAttribute<std::vector<int>>("NumTrkPt500").at(pvIndex));
    m_jetNumTrkPt1000->push_back(jet->getAttribute<std::vector<int>>("NumTrkPt1000").at(pvIndex));

    m_jetSumTrkPt500->push_back((jet->getAttribute<std::vector<float>>("SumPtTrkPt500").at(pvIndex)));
    float SumTrkPt500_vector_at_pv = jet->getAttribute<std::vector<float>>("SumPtTrkPt500").at(pvIndex);
    std::vector<float> SumTrkPt500_vector = jet->getAttribute<std::vector<float>>("SumPtTrkPt500");
    int isNotPV = 0;
    for (float item: SumTrkPt500_vector){
        if (item > SumTrkPt500_vector_at_pv){
            isNotPV = 1;
            break;
        }
    }
    isNotPVJet->push_back(isNotPV);
    
    // Following version missing in the derivation
    //m_jetSumTrkPt1000->push_back(jet->getAttribute<std::vector<float>>("SumPtTrkPt1000").at(pvIndex));
    
    //No track width in this derivation? Try another?
    m_jetTrackWidthPt1000->push_back(jet->getAttribute<std::vector<float>>("TrackWidthPt1000"));

    /*ANA_MSG_INFO("Jet: " << counter_jet << " with pT: " << (jet->pt())
                         << " has nTrack(500): " 
                         << jet->getAttribute<std::vector<int>>("NumTrkPt500").at(pvIndex)
                         << ", nTrack(1000): " 
                         << jet->getAttribute<std::vector<int>>("NumTrkPt1000").at(pvIndex)                         
                         << ", track sum Pt(500): "
                         << (jet->getAttribute<std::vector<float>>("SumPtTrkPt500").at(pvIndex)));
    */
    /////////////////////////////////////////
    //
    // Jet Contituents.
    //
    /////////////////////////////////////////

    counter_part = 0;
    const xAOD::JetConstituentVector cons = jet->getConstituents();
    for (auto cluster_itr : cons){
        //ANA_MSG_INFO ("execute(): processing event: " << m_runNumber << " eventnumber: " << m_eventNumber << " for jet number " << counter_jet <<" in particle number: "<<counter_part);
        
        //Compute delta R between consitutent and Jet
        Double_t deta = cluster_itr->eta() - jet->eta ();
        Double_t dphi = TVector2::Phi_mpi_pi(cluster_itr->phi() - jet->phi());
        Double_t dR   = TMath::Sqrt( deta*deta+dphi*dphi );
        Double_t en   = cluster_itr->e();
        
        // These two are added for the sake of junipr: need lower border
        //if(dR < 0.05){ continue;}
        if(en < 1){ continue;}
        
        //hist ("h_deltaR")->Fill (dR);
        partDeltaR->push_back(dR);
        
        
        partE->push_back(cluster_itr->e());
    	partPt->push_back(cluster_itr->pt());
        partPx->push_back(cluster_itr->px());
        partPy->push_back(cluster_itr->py());
        partPz->push_back(cluster_itr->pz());
    	partEta->push_back(cluster_itr->eta());
    	partPhi->push_back(cluster_itr->phi());
        partMass->push_back(cluster_itr->m());
        partJetCount->push_back(counter_jet);
        partRunNumber->push_back(m_runNumber);
        partEventNumber->push_back(m_eventNumber);
  	/*
        ANA_MSG_INFO ("execute(): constitutent Run " <<m_runNumber<< " and eventnumber "<<m_eventNumber);
    	ANA_MSG_INFO ("execute(): constitutent jet pt = " << (cluster_itr->pt() * 0.001) << " GeV");
    	ANA_MSG_INFO ("execute(): constitutent jet eta= " << cluster_itr->eta());
    	ANA_MSG_INFO ("execute(): constitutent jet phi= " << cluster_itr->phi());
	ANA_MSG_INFO ("execute(): constitutent to jet dR= " << dR);
	*/
        counter_part ++;
    }
    m_jetNumberConstituent->push_back (counter_part); // number of constituent
    counter_jet ++;
    //Better to use this method instead?
    //ANA_MSG_INFO("For jet at counter: " << counter_jet);
    for(long unsigned int part_it = 0; part_it < partE->size(); part_it++){
      ANA_MSG_DEBUG("parton energy vector for jet: " << partE->at(part_it));
    }

  }//End Jet Loop

  /////////////////////////////////////////
  //
  // Muons.
  //
  /////////////////////////////////////////

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
  delete m_jetWidth;
  
  delete isNotPVJet;
  delete isBadJet;
  delete isBaselineJet;
  delete isSignalJet;
  delete isBJet;
  delete passJvt;
  delete passfJvt;
  delete JvtScore;
  delete fJvtScore;
  delete btag_weight;

  delete m_jetNumTrkPt500;
  delete m_jetNumTrkPt1000;
  delete m_jetSumTrkPt500;
  delete m_jetSumTrkPt1000;
  delete m_jetTrackWidthPt500;
  delete m_jetTrackWidthPt1000;
  delete partonID;
  delete isTruthQuark;
  delete m_jetEMFrac;
  delete m_jetHECFrac;
  delete m_jetNumberConstituent;

  delete partE;
  delete partPt;
  delete partPx;
  delete partPy;
  delete partPz;
  delete partPhi;
  delete partEta;
  delete partMass;
  delete partJetCount;
  delete partDeltaR;
  delete partRunNumber;
  delete partEventNumber;
}
