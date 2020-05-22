#############################################################################
#
# Specific_Set2_Parameters.py
#
# Store usefull list and functions
#
# Author -- Maxence Draguet & Aaron O'Neill (21/05/2020)
# Adapted from Aaron O'Neill: https://github.com/aponeill89/MLForQgTaggingAndRPVSUSY/
#
#############################################################################

#List of cleaning variables to get good events.
#For now make these lists but form a cut and clean dict eventually.

#The drop operation removes rows which meet the condition
#hence the logical inversion from the standard analysis.

#cleaning_dict = [name : cut type]
#Tried to come up with an idea to pass the conditions cleverly, failed for now.
nominal_cleaning_vars = [
                         'hasBadJet',#  : == 1,
                         'hasBadMuon',# : == 1,
                         'hasCosmic',#  : == 1,
                         'passJVT',#    : == 0,
                        ]

common_cleaning_vars  = [
                         'numPrimaryVertices',#  : == 0, 
                         'GenFiltHT',#           : < 600, 
                         'PVnumTrk',#            : < 2             
                        ]

qg_tagging_vars       = [
                         'jetPt', 
                         'jetEta', 
                         'jetPhi', 
                         'jetMass',
                         'jetEnergy', 
                         'jetEMFrac', 
                         'jetHECFrac',
                         'jetChFrac', 
                         'jetNumTrkPt500', 
                         'jetNumTrkPt1000', 
                         'jetTrackWidthPt500', 
                         'jetTrackWidthPt1000',
                         'jetSumTrkPt500',
                         'jetSumTrkPt1000', 
                         'partonIDs', 
                         'BDTScore',
                         'isTruthQuark', 
                         'isBDTQuark', 
                         'isnTrkQuark'
                        ]

plot_xlabels          = {
                         'jetPt'                 : '$p_{\mathrm{T}}$ [GeV]', 
                         'jetEta'                : '$\eta$', 
                         'jetPhi'                : '$\phi$', 
                         'jetMass'               : '$M_{\mathrm{jet}}$ [GeV]',
                         'jetEnergy'             : '$E_{\mathrm{jet}}$ [GeV]', 
                         'jetEMFrac'             : 'EM Fraction', 
                         'jetHECFrac'            : 'HEC Fraction',
                         'jetChFrac'             : r'$\frac{ \sum_{i \in \mathrm{tracks}} p_{\mathrm{T ,track}}^{i} }{ p_{\mathrm{T, jet}} }$', 
                         'jetNumTrkPt500'        : '$n_{\mathrm{track}}^{500}$', 
                         'jetNumTrkPt1000'       : '$n_{\mathrm{track}}^{1000}$',
                         'jetTrackWidthPt500'    : '$w_{\mathrm{track}}^{500}$', 
                         'jetTrackWidthPt1000'   : '$w_{\mathrm{track}}^{1000}$',
                         'jetSumTrkPt500'        : '$\sum_{i \in \mathrm{tracks}} p_{\mathrm{T ,track}}^{500, i}$',
                         'jetSumTrkPt1000'       : '$\sum_{i \in \mathrm{tracks}} p_{\mathrm{T ,track}}^{500, i}$', 
                         'partonIDs'             : 'Parton ID', 
                         'BDTScore'              : 'BDT Score',
                         'isTruthQuark'          : 'Truth Quark', 
                         'isBDTQuark'            : 'BDT Quark', 
                         'isnTrkQuark'           : '$n_{\mathrm{track}}$ Quark'
                        }

plot_xranges          = {
                         'jetPt'                 : [0.0, 2800.0], 
                         'jetEta'                : [-3.0, 3.0], 
                         'jetPhi'                : [-3.5, 3.5], 
                         'jetMass'               : [0.0, 500.0],
                         'jetEnergy'             : [0.0, 4000.0], 
                         'jetEMFrac'             : [-1.0, 1.5], 
                         'jetHECFrac'            : [-1.0, 1.5],
                         'jetChFrac'             : [-1.0, 100], 
                         'jetNumTrkPt500'        : [0.0, 90.0], 
                         'jetNumTrkPt1000'       : [0.0, 90.0], 
                         'jetTrackWidthPt500'    : [0.0, 0.6], 
                         'jetTrackWidthPt1000'   : [0.0, 0.6],
                         'jetSumTrkPt500'        : [0.0, 4000],
                         'jetSumTrkPt1000'       : [0.0, 4000], 
                         'partonIDs'             : [0, 21], 
                         'BDTScore'              : [-1.0, 1.0],
                         'isTruthQuark'          : [0, 1], 
                         'isBDTQuark'            : [0, 1],
                         'isnTrkQuark'           : [0, 1]
                        }

plot_xbins          = {
                         'jetPt'                 : 56, 
                         'jetEta'                : 60, 
                         'jetPhi'                : 14, 
                         'jetMass'               : 50,
                         'jetEnergy'             : 120, 
                         'jetEMFrac'             : 25, 
                         'jetHECFrac'            : 25,
                         'jetChFrac'             : 101, 
                         'jetNumTrkPt500'        : 90, 
                         'jetNumTrkPt1000'       : 90, 
                         'jetTrackWidthPt500'    : 12, 
                         'jetTrackWidthPt1000'   : 12,
                         'jetSumTrkPt500'        : 1000,
                         'jetSumTrkPt1000'       : 1000, 
                         'partonIDs'             : 21, 
                         'BDTScore'              : 20,
                         'isTruthQuark'          : 2, 
                         'isBDTQuark'            : 2, 
                         'isnTrkQuark'           : 2
                        }
