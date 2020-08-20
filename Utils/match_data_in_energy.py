#############################################################################
#
# match_data_in_energy.py
#
# A data loader for the h5 files of the last dataset (for BDT/NN)
#
# Data from:  /data/atlas/atlasdata3/mdraguet/Set4/junipr/
# Data produced by Maxence Draguet (from GranularGatherer and processed by GranularTransformar).
#
# Author -- Maxence Draguet (16/08/2020)
#
#############################################################################
import os
import sys
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#pd.set_option('display.max_rows', None)

import Specific_Set4_Parameters
"""
# ESUB 1 GeV
save_path = "/data/atlas/atlasdata3/mdraguet/Set4_dijet_ttbar_esub1gev_matched_Completely_H5/"
quark_input_file = "/data/atlas/atlasdata3/mdraguet/Set4_dijet_ttbar_esub1gev_matched_H5/HF/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_jets.h5"
gluon_input_file = "/data/atlas/atlasdata3/mdraguet/Set4_dijet_ttbar_esub1gev_matched_H5/HF/mc16_13TeV_364700_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ_0_to_12WithSW_deriv_DAOD_JETM6_jets.h5"
"""

# ESUB 1/2 GeV
save_path = "/data/atlas/atlasdata3/mdraguet/Set4_dijet_ttbar_esub05gev_matched_Completely_H5/"
quark_input_file = "/data/atlas/atlasdata3/mdraguet/Set4_dijet_ttbar_esub05gev_H5/HF/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_jets.h5"
gluon_input_file = "/data/atlas/atlasdata3/mdraguet/Set4_dijet_ttbar_esub05gev_H5/HF/mc16_13TeV_364700_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ_0_to_12WithSW_deriv_DAOD_JETM6_jets.h5"


quark_file = pd.HDFStore(quark_input_file, "r")
gluon_file = pd.HDFStore(gluon_input_file, "r")

store_quark_keys = quark_file.keys()
store_gluon_keys = gluon_file.keys()

quark_data_input = pd.DataFrame()
gluon_data_input = pd.DataFrame()

count = 0
for key in store_quark_keys:
    count += 1
    print('Count: {0}, key: {1}'.format(count, key))
    quark_data_input = quark_data_input.append(quark_file[key])

count = 0
for key in store_gluon_keys:
    count += 1
    print('Count: {0}, key: {1}'.format(count, key))
    gluon_data_input = gluon_data_input.append(gluon_file[key])

quark_file.close()
gluon_file.close()

bins_considered = np.arange(4001)
quark_data_input['jetE_binned'] = np.floor(quark_data_input['jetE'])
gluon_data_input['jetE_binned'] = np.floor(gluon_data_input['jetE'])
resulting_quark_data_input = pd.DataFrame({i[0]: pd.Series(dtype=i[1]) for i in quark_data_input.dtypes.iteritems()}, columns=quark_data_input.dtypes.index)
resulting_gluon_data_input = pd.DataFrame({i[0]: pd.Series(dtype=i[1]) for i in gluon_data_input.dtypes.iteritems()}, columns=gluon_data_input.dtypes.index)



for bin_value in bins_considered:
    if bin_value == 4000:
        quark_data_at_bin = quark_data_input[quark_data_input['jetE_binned'] >= float(bin_value)]
        gluon_data_at_bin = gluon_data_input[gluon_data_input['jetE_binned'] >= float(bin_value)]
    else:
        quark_data_at_bin = quark_data_input[quark_data_input['jetE_binned'] == float(bin_value)]
        gluon_data_at_bin = gluon_data_input[gluon_data_input['jetE_binned'] == float(bin_value)]
    quark_number_event_found = len(quark_data_at_bin)
    gluon_number_event_found = len(gluon_data_at_bin)

    min_value_at_bin = min(quark_number_event_found, gluon_number_event_found)
    if min_value_at_bin == 0:
        print("One of the dataset is empty at bin {}".format(bin_value))
        continue
    if min_value_at_bin == quark_number_event_found:
        gluon_data_at_bin = gluon_data_at_bin[:min_value_at_bin]
    else:
        quark_data_at_bin = quark_data_at_bin[:min_value_at_bin]
    resulting_quark_data_input = resulting_quark_data_input.append(quark_data_at_bin, ignore_index=True)
    resulting_gluon_data_input = resulting_gluon_data_input.append(gluon_data_at_bin, ignore_index=True)

# Drop the discretised column
resulting_quark_data_input.drop(['jetE_binned'], axis = 1, inplace=True)
resulting_gluon_data_input.drop(['jetE_binned'], axis = 1, inplace=True)

print("Number of events in each dataset: ", len(resulting_quark_data_input))
os.makedirs(save_path, exist_ok=True)
resulting_quark_data_input.to_hdf(os.path.join(save_path, 'quark.h5'), key = "quark")
resulting_gluon_data_input.to_hdf(os.path.join(save_path, 'gluon.h5'), key = "gluon")

ttbar_path = os.path.join(save_path, 'Diagnostic_ttbar/')
dijet_path = os.path.join(save_path, 'Diagnostic_dijet/')
os.makedirs(ttbar_path, exist_ok=True)
os.makedirs(dijet_path, exist_ok=True)

vars = Specific_Set4_Parameters.qg_jet_vars

for p, plot_name in enumerate(vars):
    if plot_name in Specific_Set4_Parameters.skip_hist:
        continue
    print(plot_name)
    fig = plt.figure(p)
    ax = fig.add_subplot(1, 1, 1)
    resulting_quark_data_input.hist(column=plot_name,
                                    bins = Specific_Set4_Parameters.plot_xbins[plot_name],
                                    range = Specific_Set4_Parameters.plot_xranges[plot_name],
                                    ax = ax)
    ax.set_title(plot_name)
    ax.set_xlabel(Specific_Set4_Parameters.plot_xlabels[plot_name])
    ax.set_ylabel('Events')
    fig.savefig(os.path.join(ttbar_path, plot_name + '.png'), dpi=300, format='png', bbox_inches='tight')
    if plot_name in Specific_Set4_Parameters.log_hist:
        ax.set_yscale('log')
        fig.savefig(os.path.join(ttbar_path, plot_name + '_log.png'), dpi=300, format='png', bbox_inches='tight')
    plt.close()


for p, plot_name in enumerate(vars):
    if plot_name in Specific_Set4_Parameters.skip_hist:
        continue
    print(plot_name)
    #if plot_name == 'jetNumTrkPt500':
    #print(df[plot_name])
    fig = plt.figure(p)
    ax = fig.add_subplot(1, 1, 1)
    resulting_gluon_data_input.hist(column=plot_name,
            bins = Specific_Set4_Parameters.plot_xbins[plot_name],
            range = Specific_Set4_Parameters.plot_xranges[plot_name],
            ax = ax)
    ax.set_title(plot_name)
    ax.set_xlabel(Specific_Set4_Parameters.plot_xlabels[plot_name])
    ax.set_ylabel('Events')
    fig.savefig(os.path.join(dijet_path, plot_name + '.png'), dpi=300, format='png', bbox_inches='tight')
    if plot_name in Specific_Set4_Parameters.log_hist:
        ax.set_yscale('log')
        fig.savefig(os.path.join(dijet_path, plot_name  + '_log.png'), dpi=300, format='png', bbox_inches='tight')
    plt.close()









