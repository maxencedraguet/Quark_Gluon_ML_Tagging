 #############################################################################
#
# Specific_Set4_Parameters.py
#
# Store usefull list and functions
#
# Author -- Maxence Draguet (26/06/2020)
#
#############################################################################

#List of cleaning variables to get good events.
#For now make these lists but form a cut and clean dict eventually.

#The drop operation removes rows which meet the condition
#hence the logical inversion from the standard analysis.

#cleaning_dict = [name : cut type]
#Tried to come up with an idea to pass the conditions cleverly, failed for now.
import numpy as np

"""
nominal_cleaning_vars = [
                         'hasBadJet',#  : == 1,
                         'hasBadMuon',# : == 1,
                         'hasCosmic',#  : == 1,
                         'passJVT',#    : == 0,
                        ]
"""
"""
common_cleaning_vars  = [
                         'numPrimaryVertices',#  : == 0, 
                         'GenFiltHT',#           : < 600, 
                         'PVnumTrk',#            : < 2             
                        ]
"""
vars_convert_MeV_to_GeV_constituent = [
                                       'constituentE',
                                       'constituentPt',
                                       'constituentPx',
                                       'constituentPy',
                                       'constituentPz',
                                       'constituentMass'
                                      ]
vars_convert_MeV_to_GeV_jet = [
                               'jetPt',
                               'jetMass',
                               'jetE'
                              ]

qg_constituent_vars   = [
                         'constituentE',
                         'constituentPt',
                         'constituentPx',
                         'constituentPy',
                         'constituentPz',
                         'constituentEta',
                         'constituentPhi',
                         'constituentMass',
                         'constituentDeltaRtoJet',
                         'constituentJet',
                         'constituentRunNumber',
                         'constituentEventNumber'
                        ]


qg_jet_vars           = [
                         'jetPt', 
                         'jetEta', 
                         'jetPhi', 
                         'jetMass',
                         'jetE',
                         'jetWidth',
                         'jetEMFrac',
                         #'jetHECFrac',      # still not up
                         'jetChFrac',        # up
                         'jetNumTrkPt500',
                         'jetNumTrkPt1000',
                         #'jetTrackWidthPt500',     # not in derivation
                         'jetTrackWidthPt1000',     # but this one is for some reason
                         'jetSumTrkPt500',
                         #'jetSumTrkPt1000', # weirdly not up yet
                         'partonID',         # this is supposedly the truth info of the jet ID. Should get a way of seeing if it's quark or gluon
                         'isTruthQuark',      # same as above but regroups quark value to 1 (pID between 1 and 6), gluon to 0 (pID 21) and other to -1
                         'jetNumberConstituent'
                        ]
""",
    'jetEMFrac',
    'jetHECFrac',
    'jetChFrac',
    'jetNumTrkPt500',
    'jetNumTrkPt1000',
    'jetTrackWidthPt500',
    'jetTrackWidthPt1000',
    'jetSumTrkPt500',
    'jetSumTrkPt1000',
    'partonID',
    'BDTScore',
    'isTruthQuark',
    'isBDTQuark',
    'isnTrkQuark'"""

"""
    # What follows is a list of jet quality cuts that should properly be used at the algo level
    'isBadJet',
    'isBaselineJet',
    'isSignalJet',
    'isBJet',
    'passJvt',
    'passfJvt',
    'JvtScore',
    'fJvtScore',
    'btag_weight'
    
    """

plot_xlabels          = {
                         'jetPt'                 : '$p_{\mathrm{T}}$ [GeV]',
                         'jetEta'                : '$\eta$', 
                         'jetPhi'                : '$\phi$', 
                         'jetMass'               : '$M_{\mathrm{jet}}$ [GeV]',
                         'jetE'                  : '$E_{\mathrm{jet}}$ [GeV]',
                         'jetEMFrac'             : 'EM Fraction', 
                         'jetHECFrac'            : 'HEC Fraction',
                         'jetChFrac'             : r'$\frac{ \sum_{i \in \mathrm{tracks}} p_{\mathrm{T ,track}}^{i} }{ p_{\mathrm{T, jet}} }$', 
                         'jetNumTrkPt500'        : '$n_{\mathrm{track}}^{500}$', 
                         'jetNumTrkPt1000'       : '$n_{\mathrm{track}}^{1000}$',
                         'jetTrackWidthPt500'    : '$w_{\mathrm{track}}^{500}$', 
                         'jetTrackWidthPt1000'   : '$w_{\mathrm{track}}^{1000}$',
                         'jetSumTrkPt500'        : '$\sum_{i \in \mathrm{tracks}} p_{\mathrm{T ,track}}^{500, i}$',
                         'jetSumTrkPt1000'       : '$\sum_{i \in \mathrm{tracks}} p_{\mathrm{T ,track}}^{500, i}$', 
                         'partonID'             : 'Parton ID',
                         'BDTScore'              : 'BDT Score',
                         'isTruthQuark'          : 'Truth Quark', 
                         'isBDTQuark'            : 'BDT Quark', 
                         'isnTrkQuark'           : '$n_{\mathrm{track}}$ Quark',
                         'jetWidth'              : 'Width of the jet',
                         'jetNumberConstituent'  : 'Number of constituents per jet (algorithm)',
                        ##### Constituents
                        'constituentPt'          : '$p_{\mathrm{T}}$ [GeV]',
                        'constituentPx'          : '$p_{x}$ [GeV]',
                        'constituentPy'          : '$p_{y}$ [GeV]',
                        'constituentPz'          : '$p_{z}$ [GeV]',
                        'constituentEta'         : '$\eta$',
                        'constituentPhi'         : '$\phi$',
                        'constituentMass'        : '$M_{\mathrm{cell}}$ [GeV]',
                        'constituentE'           : '$E_{\mathrm{cell}}$ [GeV]',
                        'constituentDeltaRtoJet' : '$\Delta_R$ Jet-Constituent [GeV]',
                        ##### Additional
                        'differenceEnergy'       : 'Jet Energy - $\Sigma$ constituent Energy [GeV]',
                        'differencePx'           : 'Jet $p_x$ - $\Sigma$ constituent $p_x$ [GeV]',
                        'differencePy'           : 'Jet $p_y$ - $\Sigma$ constituent $p_y$ [GeV]',
                        'differencePx_computed'  : 'Jet $p_x$ - $\Sigma$ constituent $p_x$ (from angles) [GeV]',
                        'differencePy_computed'  : 'Jet $p_y$ - $\Sigma$ constituent $p_y$ (from angles) [GeV]',
                        'CounterElem'            : 'Number of constituents per jet'
                        
                        
                        }

plot_xranges          = {
                         'jetPt'                 : [0.0, 2000.0],
                         'jetEta'                : [-3.0, 3.0], 
                         'jetPhi'                : [-3.5, 3.5], 
                         'jetMass'               : [0.0, 500.0],
                         'jetE'                  : [0.0, 6000.0],
                         'jetEMFrac'             : [-1.0, 1.5], 
                         'jetHECFrac'            : [-1.0, 1.5],
                         'jetChFrac'             : [-1.0, 100], 
                         'jetNumTrkPt500'        : [0.0, 90.0], 
                         'jetNumTrkPt1000'       : [0.0, 90.0], 
                         'jetTrackWidthPt500'    : [0.0, 0.6], 
                         'jetTrackWidthPt1000'   : [0.0, 0.6],
                         'jetSumTrkPt500'        : [0.0, 4000],
                         'jetSumTrkPt1000'       : [0.0, 4000], 
                         'partonID'             : [0, 21],
                         'BDTScore'              : [-1.0, 1.0],
                         'isTruthQuark'          : [-1, 1],
                         'isBDTQuark'            : [0, 1],
                         'isnTrkQuark'           : [0, 1],
                         'jetWidth'              : [0, 1],
                         'jetNumberConstituent'  : [0, 50],
                             ####
                         'constituentPt'         : [-10.0, 400.0],
                         'constituentPx'         : [-400.0, 400.0],
                         'constituentPy'         : [-400.0, 400.0],
                         'constituentPz'         : [-400.0, 400.0],
                         'constituentEta'        : [-3.0, 3.0],
                         'constituentPhi'        : [-3.5, 3.5],
                         'constituentMass'       : [0.0, 1.0],
                         'constituentE'          : [-100.0, 700.0],
                         'constituentDeltaRtoJet': [0.0, 0.7],
                         ##### Additional
                         'differenceEnergy'       : [-100.0, 2000.0],
                         'differencePx'           : [-300.0, 300.0],
                         'differencePy'           : [-300.0, 300.0],
                         'differencePx_computed'  : [-300.0, 300.0],
                         'differencePy_computed'  : [-300.0, 300.0],
                         'CounterElem'            : [0, 50]
                             
                        }

plot_xbins          = {
                         'jetPt'                 : 56,
                         'jetEta'                : 60, 
                         'jetPhi'                : 1000,
                         'jetMass'               : 50,
                         'jetE'                  : 120,
                         'jetEMFrac'             : 25, 
                         'jetHECFrac'            : 25,
                         'jetChFrac'             : 101, 
                         'jetNumTrkPt500'        : 90, 
                         'jetNumTrkPt1000'       : 90, 
                         'jetTrackWidthPt500'    : 12, 
                         'jetTrackWidthPt1000'   : 12,
                         'jetSumTrkPt500'        : 100,
                         'jetSumTrkPt1000'       : 100,
                         'partonID'             : 21,
                         'BDTScore'              : 20,
                         'isTruthQuark'          : 3,
                         'isBDTQuark'            : 2, 
                         'isnTrkQuark'           : 2,
                         'jetWidth'              : 200,
                         'jetNumberConstituent'  : 51,
                             ###
                         'constituentPt'         : 150,
                         'constituentPx'         : 150,
                         'constituentPy'         : 150,
                         'constituentPz'         : 150,
                         'constituentEta'        : 100,
                         'constituentPhi'        : 1000,
                         'constituentMass'       : 100,
                         'constituentE'          : 120,
                         'constituentDeltaRtoJet': 200,
                         ##### Additional
                         'differenceEnergy'       : 100,
                         'differencePx'           : 100,
                         'differencePy'           : 100,
                         'differencePx_computed'  : 100,
                         'differencePy_computed'  : 100,
                         'CounterElem'            : 51
                        
                     }

skip_hist = ['constituentJet', 'constituentRunNumber', 'constituentEventNumber', 'jetTrackWidthPt1000']

log_hist = ['jetNumberConstituent', 'jetE', 'constituentE', 'jetPt', 'constituentPt', 'constituentPx', 'constituentPy', 'constituentPz', 'differenceEnergy', 'differencePx', 'differencePy', 'differencePx_computed', 'differencePy_computed']



