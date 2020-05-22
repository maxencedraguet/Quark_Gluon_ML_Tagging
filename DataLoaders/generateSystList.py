#############################################################################
#
# generateSystList.py
#
# Automatically generate the systematics lists for q/g tagging.
#
# Author -- Aaron O'Neill (01/05/2020)
# email  -- aaron.aoneill@physics.ox.ac.uk
#
#############################################################################

systematicDict = {}

systematicDict["JES0"] = [
                          "JET_BJES_Response__1up",
                          "JET_BJES_Response__1down",
                          "JET_EffectiveNP_1__1up",
                          "JET_EffectiveNP_1__1down",
                          "JET_EffectiveNP_2__1up",
                          "JET_EffectiveNP_2__1down",
                          "JET_EffectiveNP_3__1up",
                          "JET_EffectiveNP_3__1down",
                          "JET_EffectiveNP_4__1up",
                          "JET_EffectiveNP_4__1down",
                          "JET_EffectiveNP_5__1up",
                          "JET_EffectiveNP_5__1down",
                          "JET_EffectiveNP_6restTerm__1up",
                          "JET_EffectiveNP_6restTerm__1down",
                          "JET_EtaIntercalibration_Modelling__1up",
                          "JET_EtaIntercalibration_Modelling__1down",
                          "JET_EtaIntercalibration_TotalStat__1up",
                          "JET_EtaIntercalibration_TotalStat__1down",
                          "JET_Flavor_Composition__1up",
                          "JET_Flavor_Composition__1down",
                          "JET_Flavor_Response__1up",
                          "JET_Flavor_Response__1down",
                          "JET_Pileup_OffsetMu__1up",
                          "JET_Pileup_OffsetMu__1down",
                          "JET_Pileup_OffsetNPV__1up",
                          "JET_Pileup_OffsetNPV__1down",
                          "JET_Pileup_PtTerm__1up",
                          "JET_Pileup_PtTerm__1down",
                          "JET_Pileup_RhoTopology__1up",
                          "JET_Pileup_RhoTopology__1down",
                          "JET_PunchThrough_MC16__1up",
                          "JET_PunchThrough_MC16__1down",
                          "JET_PunchThrough_MC15__1up",
                          "JET_PunchThrough_MC15__1down",
                          "JET_SingleParticle_HighPt__1up",
                          "JET_SingleParticle_HighPt__1down",
                          "JET_GroupedNP_1__1up",
                          "JET_GroupedNP_1__1down",
                          "JET_JER_DataVsMC_MC16__1up",
                          "JET_JER_DataVsMC_MC16__1down",
                          "JET_JER_EffectiveNP_1__1up",
                          "JET_JER_EffectiveNP_1__1down",
                          "JET_JER_EffectiveNP_2__1up",
                          "JET_JER_EffectiveNP_2__1down",
                          "JET_JER_EffectiveNP_3__1up",
                          "JET_JER_EffectiveNP_3__1down",
                          "JET_JER_EffectiveNP_4__1up",
                          "JET_JER_EffectiveNP_4__1down",
                          "JET_JER_EffectiveNP_5__1up",
                          "JET_JER_EffectiveNP_5__1down",
                          "JET_JER_EffectiveNP_6__1up",
                          "JET_JER_EffectiveNP_6__1down",
                          "JET_JER_EffectiveNP_7restTerm__1up",
                          "JET_JER_EffectiveNP_7restTerm__1down",
                          "JET_JvtEfficiency__1down",
                          "JET_JvtEfficiency__1up"
                          ]

systematicDict["JES1"] = [ # For Fall 2018 strong reduction
                          "JET_GroupedNP_1__1up",
                          "JET_GroupedNP_1__1down",
                          "JET_GroupedNP_2__1up",
                          "JET_GroupedNP_2__1down",
                          "JET_GroupedNP_3__1up",
                          "JET_GroupedNP_3__1down",
                          "JET_Comb_Baseline_Kin__1up",
                          "JET_Comb_Baseline_Kin__1down",
                          "JET_Comb_Modelling_Kin__1up",
                          "JET_Comb_Modelling_Kin__1down",
                          "JET_Comb_TotalStat_Kin__1up",
                          "JET_Comb_TotalStat_Kin__1down",
                          "JET_Comb_Tracking_Kin__1up",
                          "JET_Comb_Tracking_Kin__1down",
                          "JET_EtaIntercalibration_NonClosure_highE__1up",
                          "JET_EtaIntercalibration_NonClosure_highE__1down",
                          "JET_EtaIntercalibration_NonClosure_negEta__1up",
                          "JET_EtaIntercalibration_NonClosure_negEta__1down",
                          "JET_EtaIntercalibration_NonClosure_posEta__1up",
                          "JET_EtaIntercalibration_NonClosure_posEta__1down",
                          "JET_Flavor_Response__1up",
                          "JET_Flavor_Response__1down",
                          "JET_JER_DataVsMC_MC16__1up",
                          "JET_JER_DataVsMC_MC16__1down",
                          "JET_JER_EffectiveNP_1__1up",
                          "JET_JER_EffectiveNP_1__1down",
                          "JET_JER_EffectiveNP_2__1up",
                          "JET_JER_EffectiveNP_2__1down",
                          "JET_JER_EffectiveNP_3__1up",
                          "JET_JER_EffectiveNP_3__1down",
                          "JET_JER_EffectiveNP_4__1up",
                          "JET_JER_EffectiveNP_4__1down",
                          "JET_JER_EffectiveNP_5__1up",
                          "JET_JER_EffectiveNP_5__1down",
                          "JET_JER_EffectiveNP_6__1up",
                          "JET_JER_EffectiveNP_6__1down",
                          "JET_JER_EffectiveNP_7restTerm__1up",
                          "JET_JER_EffectiveNP_7restTerm__1down",
                          "JET_JvtEfficiency__1down",
                          "JET_JvtEfficiency__1up"
                          ]

systematicDict["JES2"] = [ # For Fall 2018 category reduction
                          "JET_EffectiveNP_Statistical1__1up",
                          "JET_EffectiveNP_Statistical1__1down",
                          "JET_EffectiveNP_Statistical2__1up",
                          "JET_EffectiveNP_Statistical2__1down",
                          "JET_EffectiveNP_Statistical3__1up",
                          "JET_EffectiveNP_Statistical3__1down",
                          "JET_EffectiveNP_Statistical4__1up",
                          "JET_EffectiveNP_Statistical4__1down",
                          "JET_EffectiveNP_Statistical5__1up",
                          "JET_EffectiveNP_Statistical5__1down",
                          "JET_EffectiveNP_Statistical6__1up",
                          "JET_EffectiveNP_Statistical6__1down",
                          "JET_EffectiveNP_Modelling1__1up",
                          "JET_EffectiveNP_Modelling1__1down",
                          "JET_EffectiveNP_Modelling2__1up",
                          "JET_EffectiveNP_Modelling2__1down",
                          "JET_EffectiveNP_Modelling3__1up",
                          "JET_EffectiveNP_Modelling3__1down",
                          "JET_EffectiveNP_Modelling4__1up",
                          "JET_EffectiveNP_Modelling4__1down",
                          "JET_EffectiveNP_Detector1__1up",
                          "JET_EffectiveNP_Detector1__1down",
                          "JET_EffectiveNP_Detector2__1up",
                          "JET_EffectiveNP_Detector2__1down",
                          "JET_EffectiveNP_Mixed1__1up",
                          "JET_EffectiveNP_Mixed1__1down",
                          "JET_EffectiveNP_Mixed2__1up",
                          "JET_EffectiveNP_Mixed2__1down",
                          "JET_EffectiveNP_Mixed3__1up",
                          "JET_EffectiveNP_Mixed3__1down",
                          "JET_EtaIntercalibration_Modelling__1up",
                          "JET_EtaIntercalibration_Modelling__1down",
                          "JET_EtaIntercalibration_TotalStat__1up",
                          "JET_EtaIntercalibration_TotalStat__1down",
                          "JET_EtaIntercalibration_NonClosure_highE__1up",
                          "JET_EtaIntercalibration_NonClosure_highE__1down",
                          "JET_EtaIntercalibration_NonClosure_negEta__1up",
                          "JET_EtaIntercalibration_NonClosure_negEta__1down",
                          "JET_EtaIntercalibration_NonClosure_posEta__1up",
                          "JET_EtaIntercalibration_NonClosure_posEta__1down",
                          "JET_SingleParticle_HighPt__1up",
                          "JET_SingleParticle_HighPt__1down",
                          "JET_Pileup_OffsetMu__1up",
                          "JET_Pileup_OffsetMu__1down",
                          "JET_Pileup_OffsetNPV__1up",
                          "JET_Pileup_OffsetNPV__1down",
                          "JET_Pileup_PtTerm__1up",
                          "JET_Pileup_PtTerm__1down",
                          "JET_Pileup_RhoTopology__1up",
                          "JET_Pileup_RhoTopology__1down",
                          "JET_Flavor_Composition__1up",
                          "JET_Flavor_Composition__1down",
                          "JET_Flavor_Response__1up",
                          "JET_Flavor_Response__1down",
                          "JET_BJES_Response__1up",
                          "JET_BJES_Response__1down",
                          #MC16 required for signal samples
                          #"JET_PunchThrough_AFII__1up",
                          #"JET_PunchThrough_AFII__1down",
                          #"JET_JER_DataVsMC_AFII__1up",
                          #"JET_JER_DataVsMC_AFII__1down",
                          #MC16 names required for background
                          "JET_PunchThrough_MC16__1up",
                          "JET_PunchThrough_MC16__1down",
                          "JET_JER_DataVsMC_MC16__1up",
                          "JET_JER_DataVsMC_MC16__1down",
                          "JET_JER_EffectiveNP_1__1up",
                          "JET_JER_EffectiveNP_1__1down",
                          "JET_JER_EffectiveNP_2__1up",
                          "JET_JER_EffectiveNP_2__1down",
                          "JET_JER_EffectiveNP_3__1up",
                          "JET_JER_EffectiveNP_3__1down",
                          "JET_JER_EffectiveNP_4__1up",
                          "JET_JER_EffectiveNP_4__1down",
                          "JET_JER_EffectiveNP_5__1up",
                          "JET_JER_EffectiveNP_5__1down",
                          "JET_JER_EffectiveNP_6__1up",
                          "JET_JER_EffectiveNP_6__1down",
                          "JET_JER_EffectiveNP_7restTerm__1up",
                          "JET_JER_EffectiveNP_7restTerm__1down",
                          "JET_fJvtEfficiency__1down",
                          "JET_fJvtEfficiency__1up",
                          #"JET_QG_trackEfficiency",
                          #"JET_QG_trackFakes",
                          #"JET_QG_nchargedVBF",
                          #"JET_QG_nchargedExp__up",
                          #"JET_QG_nchargedExp__down",
                          #"JET_QG_nchargedME__up",
                          #"JET_QG_nchargedME__down",
                          #"JET_QG_nchargedPDF__up",
                          #"JET_QG_nchargedPDF__down"
                          ]

systematicDict["FTAG"] = [
                          "FT_EFF_B_systematics__1down",
                          "FT_EFF_B_systematics__1up",
                          "FT_EFF_C_systematics__1down",
                          "FT_EFF_C_systematics__1up",
                          "FT_EFF_Light_systematics__1down",
                          "FT_EFF_Light_systematics__1up",
                          "FT_EFF_extrapolation__1down",
                          "FT_EFF_extrapolation__1up",
                          "FT_EFF_extrapolation_from_charm__1down",
                          "FT_EFF_extrapolation_from_charm__1up"
                          ]

systematicDict["PRW"] = [
                         "PRW_DATASF__1down",
                         "PRW_DATASF__1up"
                         ]

systematicDict["MET"] = [
                         "MET_SoftTrk_ResoPara",
                         "MET_SoftTrk_ResoPerp",
                         "MET_SoftTrk_ScaleDown",
                         "MET_SoftTrk_ScaleUp"
                         ]

systematicDict["ELE"] = [
                         "EG_RESOLUTION_ALL__1down",
                         "EG_RESOLUTION_ALL__1up",
                         "EG_SCALE_ALL__1down",
                         "EG_SCALE_ALL__1up",
                         "EG_SCALE_AF2__1down",
                         "EG_SCALE_AF2__1up",
                         "EL_EFF_ChargeIDSel_TOTAL_1NPCOR_PLUS_UNCOR__1down",
                         "EL_EFF_ChargeIDSel_TOTAL_1NPCOR_PLUS_UNCOR__1up",
                         "EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR__1down",
                         "EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR__1up",
                         "EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR__1down",
                         "EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR__1up",
                         "EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR__1down",
                         "EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR__1up",
                         "EL_EFF_Trigger_TOTAL_1NPCOR_PLUS_UNCOR__1up",
                         "EL_EFF_Trigger_TOTAL_1NPCOR_PLUS_UNCOR__1down",
                         "EL_EFF_TriggerEff_TOTAL_1NPCOR_PLUS_UNCOR__1up",
                         "EL_EFF_TriggerEff_TOTAL_1NPCOR_PLUS_UNCOR__1down"
                         ]

systematicDict["MUON"] = [
                          "MUON_ID__1down",
                          "MUON_ID__1up",
                          "MUON_MS__1down",
                          "MUON_MS__1up",
                          "MUON_SAGITTA_RESBIAS__1down",
                          "MUON_SAGITTA_RESBIAS__1up",
                          "MUON_SAGITTA_RHO__1down",
                          "MUON_SAGITTA_RHO__1up",
                          "MUON_SCALE__1down",
                          "MUON_SCALE__1up",
                          "MUON_EFF_BADMUON_STAT__1down",
                          "MUON_EFF_BADMUON_STAT__1up",
                          "MUON_EFF_BADMUON_SYS__1down",
                          "MUON_EFF_BADMUON_SYS__1up",
                          "MUON_EFF_ISO_STAT__1down",
                          "MUON_EFF_ISO_STAT__1up",
                          "MUON_EFF_ISO_SYS__1down",
                          "MUON_EFF_ISO_SYS__1up",
                          "MUON_EFF_RECO_STAT__1down",
                          "MUON_EFF_RECO_STAT__1up",
                          "MUON_EFF_RECO_STAT_LOWPT__1down",
                          "MUON_EFF_RECO_STAT_LOWPT__1up",
                          "MUON_EFF_RECO_SYS__1down",
                          "MUON_EFF_RECO_SYS__1up",
                          "MUON_EFF_RECO_SYS_LOWPT__1down",
                          "MUON_EFF_RECO_SYS_LOWPT__1up",
                          "MUON_EFF_TTVA_STAT__1down",
                          "MUON_EFF_TTVA_STAT__1up",
                          "MUON_EFF_TTVA_SYS__1down",
                          "MUON_EFF_TTVA_SYS__1up",
                          "MUON_EFF_TrigStatUncertainty__1down",
                          "MUON_EFF_TrigStatUncertainty__1up",
                          "MUON_EFF_TrigSystUncertainty__1down",
                          "MUON_EFF_TrigSystUncertainty__1up"
                          ]

systematicDict["PHOT"] = [
                          "PH_EFF_ID_Uncertainty__1down",
                          "PH_EFF_ID_Uncertainty__1up",
                          "PH_EFF_ISO_Uncertainty__1down",
                          "PH_EFF_ISO_Uncertainty__1up",
                          "PH_EFF_TRIGGER_Uncertainty__1down",
                          "PH_EFF_TRIGGER_Uncertainty__1up"
                          ]

systematicDict["TAU"] = [
                         "TAUS_TRUEHADTAU_EFF_ELEOLR_TOTAL__1down",
                         "TAUS_TRUEHADTAU_EFF_ELEOLR_TOTAL__1up",
                         "TAUS_TRUEHADTAU_EFF_JETID_TOTAL__1down",
                         "TAUS_TRUEHADTAU_EFF_JETID_TOTAL__1up",
                         "TAUS_TRUEHADTAU_EFF_RECO_TOTAL__1down",
                         "TAUS_TRUEHADTAU_EFF_RECO_TOTAL__1up",
                         "TAUS_TRUEHADTAU_EFF_TES_TOTAL__1down",
                         "TAUS_TRUEHADTAU_EFF_TES_TOTAL__1up"
                         ]

systematicDict["TILE"] = [
                          "JET_TILECORR_Uncertainty__1up",
                          "JET_TILECORR_Uncertainty__1down"
                          ]

def getSystematicList(systOption, JESNPSet):
    systematicList = ["Nominal"]
    if "JES" in systOption or systOption in ["Minimal", "All"]:
        if JESNPSet == 0: systematicList += systematicDict["JES0"]
        if JESNPSet == 2: systematicList += systematicDict["JES2"] # Category reduction
        else: systematicList += systematicDict["JES1"]
            #else: systematicList += systematicDict["JES2"] # Category reduction
    if "FTAG" in systOption or systOption in ["Minimal", "All"]:
        systematicList += systematicDict["FTAG"]
    if "MET" in systOption or systOption in ["Minimal", "All"]:
        systematicList += systematicDict["MET"]
    if "PRW" in systOption or systOption in ["All"]: # Choosing this approach so that we can easily insert more custom systematic sets
        systematicList += systematicDict["PRW"]
    if "ELE" in systOption or systOption in ["All"]:
        systematicList += systematicDict["ELE"]
    if "MUON" in systOption or systOption in ["All"]:
        systematicList += systematicDict["MUON"]
    if "PHOT" in systOption or systOption in ["All"]:
        systematicList += systematicDict["PHOT"]
    if "TAU" in systOption or systOption in []:
        systematicList += systematicDict["TAU"]
    if "TILE" in systOption or systOption in ["All"]:
        systematicList += systematicDict["TILE"]
    return systematicList
