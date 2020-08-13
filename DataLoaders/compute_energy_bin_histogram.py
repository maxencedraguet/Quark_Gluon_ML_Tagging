#############################################################################
#
# compute_energu_bin_histogram_JSON.py
#
# Reads junipr json files for ttbar files to return the histogram of energy.
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


apply_limitor = True
target_label = 1            # 1 to work on the quark jets, 0 for the gluon ones.
file_limitor =    ["/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev_pure/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_test_bin_count.txt",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev_pure/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_train_bin_count.txt"]

# This is for the ttbar part II.
# Slight twist, this accumulates the formed array and save it to a single item (for the train and test separately).
#It also makes a dataframe of the number of jet per energy bin.
save_single_json = True

"""
file_path_list = [["/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part1_esub1gev/junipr/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH1_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH2_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH3_deriv_DAOD_JETM6_test.json"],
                  
                  ["/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part1_esub1gev/junipr/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH1_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH2_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH3_deriv_DAOD_JETM6_train.json"]]

file_name_list = ["mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_test", "mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_train"]
"""
file_path_list = [["/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part1_esub1gev/junipr/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH1_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH2_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH3_deriv_DAOD_JETM6_test.json"],
                  
                  ["/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part1_esub1gev/junipr/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH1_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH2_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH3_deriv_DAOD_JETM6_train.json"]]

file_name_list = ["mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_test", "mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_train"]

file_path_list = [["/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364700.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364701.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364704.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364705.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364706.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364707.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364709.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364710.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364711.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364712.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_test.json"],
                  ["/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364700.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364701.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364704.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364705.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364706.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364707.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364709.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364710.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364711.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV.364712.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128_train.json"]]

file_name_list = ["mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_test", "mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_train"]

save_path = "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev_pure_matched/"     # target for save
os.makedirs(save_path, exist_ok=True)


if apply_limitor:
    limitor_list = list()
    for item in file_limitor:
        limiting_array = np.loadtxt(item, delimiter=',', , fmt='%i')
        limitor_list.append(limiting_array)

for l in range(len(file_path_list)):
    count_energy_per_bin = np.zeros(4001).astype('int') # everything above or equal 4000 GeV is seen as a single bin
    bin = np.arange(4001)
    if apply_limitor:
        limiting_array = limitor_list[l][0,:]   # no need to take bin values, they are the same here
    file_path_sub_list = file_path_list[l]
    file_name = file_name_list[l]
    
    global_data_array = list()
    for file_path in file_path_sub_list:
        print(file_path)
        with open(file_path) as json_file:
            data_array = json.load(json_file)['JuniprJets']
            global_data_array.extend(data_array)

    # make the histogram of energy and cut based on label
    new_list = list()
    energy_list = list()
    counter = 0
    counter_limitor = 0
    for count, item in enumerate(global_data_array):
        label = item["label"]
        seed_momentum = item["seed_momentum"]
        energy = seed_momentum[0]
        if label == target_label:
            counter += 1
            if apply_limitor:
                energy_roud = np.floor(energy).astype(int)
                if energy_roud > 4000:
                    energy_roud = 4000
                if count_energy_per_bin[energy_roud] < limiting_array[energy_roud]:
                    new_list.append(item)
                    energy_list.append(energy)
                    count_energy_per_bin[energy_roud] +=1
            
            else:
                new_list.append(item)
                energy_list.append(energy)
                energy_roud = np.floor(energy).astype(int)
                if energy_roud > 4000:
                    energy_roud = 4000
                count_energy_per_bin[energy_roud] +=1

    print("In total, {} jets were found with the label {} for {}".format(counter, target_label, file_name))
    new_json = {'JuniprJets':  new_list}
    with open(os.path.join(save_path, file_name + '.json'), "w") as write_file:
        json.dump(new_json, write_file) #, indent=4)

    stacked_arrays = np.stack([count_energy_per_bin, bin], axis = 0)
    np.savetxt(os.path.join(save_path, file_name + '_bin_count.txt'), stacked_arrays, delimiter=',', fmt='%i')

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

