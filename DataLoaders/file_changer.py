#############################################################################
#
# file_changer.py
#
# reads junipr json files for quark and gluons. Turn them into pure samples
#
# Author -- Maxence Draguet (06/07/2020)
#
#############################################################################
import os
import sys
import json
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path_list = ["/data/atlas/atlasdata3/mdraguet/Set4_quark_gluon_full/junipr/mc16_13TeV_364704_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_deriv_DAOD_JETM6_test.json",
"/data/atlas/atlasdata3/mdraguet/Set4_quark_gluon_full/junipr/mc16_13TeV_364704_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_deriv_DAOD_JETM6_train.json",
"/data/atlas/atlasdata3/mdraguet/Set4_quark_gluon_full/junipr/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_test.json",
"/data/atlas/atlasdata3/mdraguet/Set4_quark_gluon_full/junipr/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_train.json"]

file_name_list = ["mc16_13TeV_364704_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_deriv_DAOD_JETM6_test.json",
"mc16_13TeV_364704_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_deriv_DAOD_JETM6_train.json",
"mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_test.json",
"mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_train.json"]

save_path = "/data/atlas/atlasdata3/mdraguet/Set4_quark_gluon_pure_unmatched/junipr/"     # target for save
os.makedirs(save_path, exist_ok=True)

target_label_list = [0, 0, 1, 1]   # 0 for gluon, 1 for quark

for l in range(len(file_path_list)):
    file_path = file_path_list[l]
    file_name = file_name_list[l]
    target_label = target_label_list[l]
    
    with open(file_path) as json_file:
        data_array = json.load(json_file)['JuniprJets']

    new_list = list()
    energy_list = list()
    counter = 0
    for item in data_array:
        label = item["label"]
        seed_momentum = item["seed_momentum"]
        energy = seed_momentum[0]
        if label == target_label:
            counter += 1
            new_list.append(item)
            energy_list.append(energy)

    print("In total, {} jets were found with the label {} in {}".format(counter, target_label, file_path))
    new_json = {'JuniprJets':  new_list}
    with open(os.path.join(save_path, file_name), "w") as write_file:
        json.dump(new_json, write_file) #, indent=4)


    df = pd.DataFrame(data=energy_list, columns=["jetE"])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    df.hist(column = "jetE",
              bins = 120,
              range = [0.0, 6000.0],
              ax = ax)
    ax.set_title("jetE")
    ax.set_xlabel('$E_{\mathrm{jet}}$ [GeV]')
    ax.set_ylabel('Count')
    fig.savefig(os.path.join(save_path, file_name + '_energy.png'), dpi=300, format='png', bbox_inches='tight')
    ax.set_yscale('log')
    fig.savefig(os.path.join(save_path, file_name + '_energy_log.png'), dpi=300, format='png', bbox_inches='tight')
    plt.close()

