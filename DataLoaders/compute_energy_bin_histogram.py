#############################################################################
#
# compute_energy_bin_histogram.py
#
# Reads junipr json  files to return the histogram of energy of the underlying data.
# Can also be used to scale samples to a given energy distribution (by cutting down extra
# data.
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
target_label = 1      # 1 to work on the quark jets, 0 for the gluon ones.

"""
# Cutting on energy of pure quark tbar
file_limitor = ["/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev_pure/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_test_bin_count.txt",
                "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev_pure/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_train_bin_count.txt"]
"""

"""
# This is for the ttbar process 1 GeV E_sub
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

"""
# Second step of the ttbar process 1 GeV E_sub
file_path_list = [["/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev_pure/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_test.json"],
                  ["/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev_pure/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_train.json"]]

file_name_list = ["mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_test",
                  "mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_train"]
                  
# Cutting on energy of pure matched gluon dijet
file_limitor = ["/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev_pure_matched/mc16_13TeV_364702_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0_to_12WithSW_deriv_DAOD_JETM6_test_bin_count.txt",
"/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev_pure_matched/mc16_13TeV_364702_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0_to_12WithSW_deriv_DAOD_JETM6_train_bin_count.txt"]
"""

"""
# This is for the dijet process, all in one step for 1 GeV E_sub
file_path_list = [["/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364700_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364701_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364702_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364703_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364704_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364705_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364706_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364707_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364709_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364710_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364711_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364712_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW_deriv_DAOD_JETM6_test.json"],
                  
                   ["/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364700_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW_deriv_DAOD_JETM6_train.json",
                    "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364701_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW_deriv_DAOD_JETM6_train.json",
                    "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364702_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_deriv_DAOD_JETM6_train.json",
                    "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364703_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_deriv_DAOD_JETM6_train.json",
                    "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364704_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_deriv_DAOD_JETM6_train.json",
                    "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364705_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_deriv_DAOD_JETM6_train.json",
                    "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364706_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW_deriv_DAOD_JETM6_train.json",
                    "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364707_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW_deriv_DAOD_JETM6_train.json",
                    "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364709_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW_deriv_DAOD_JETM6_train.json",
                    "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364710_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW_deriv_DAOD_JETM6_train.json",
                    "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364711_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW_deriv_DAOD_JETM6_train.json",
                    "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/mc16_13TeV_364712_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW_deriv_DAOD_JETM6_train.json"]]


file_name_list = ["mc16_13TeV_364702_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0_to_12WithSW_deriv_DAOD_JETM6_test",
                  "mc16_13TeV_364702_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0_to_12WithSW_deriv_DAOD_JETM6_train"]
"""
"""
# This is for the ttbar process 0.5 GeV E_sub
file_path_list = [["/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH1_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch2/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH2_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch3/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH3_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch4/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH4_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch5/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH5_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch6/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH6_deriv_DAOD_JETM6_test.json"],
                  ["/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH1_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch2/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH2_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch3/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH3_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch4/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH4_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch5/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH5_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch6/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH6_deriv_DAOD_JETM6_train.json"]]

file_name_list = ["mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_test", "mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_train"]
"""

"""
# This is for the dijet process, all in one step for 0.5 GeV E_sub
file_path_list = [[#"/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364700_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364701_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364702_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364703_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364704_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364705_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364706_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364707_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364709_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364710_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364711_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364712_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW_deriv_DAOD_JETM6_test.json"],
                  [#"/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364700_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364701_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364702_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364703_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364704_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364705_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364706_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364707_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364709_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364710_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364711_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev/mc16_13TeV_364712_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW_deriv_DAOD_JETM6_train.json"]]

file_name_list = ["mc16_13TeV_364702_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0_to_12WithSW_deriv_DAOD_JETM6_test",
                  "mc16_13TeV_364702_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0_to_12WithSW_deriv_DAOD_JETM6_train"]
                  
file_limitor = ["/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev_pure/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_test_bin_count.txt",
                "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev_pure/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_train_bin_count.txt"]
"""

# Second step of the ttbar process 0.5 GeV E_sub
file_path_list = [["/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH1_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch2/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH2_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch3/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH3_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch4/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH4_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch5/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH5_deriv_DAOD_JETM6_test.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch6/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH6_deriv_DAOD_JETM6_test.json"],
                  ["/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch1/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH1_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch2/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH2_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch3/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH3_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch4/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH4_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch5/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH5_deriv_DAOD_JETM6_train.json",
                   "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev/batch6/mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH6_deriv_DAOD_JETM6_train.json"]]

file_name_list = ["mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_test", "mc16_13TeV_410470_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_deriv_DAOD_JETM6_train"]

file_limitor = ["/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev_pure_matched/mc16_13TeV_364702_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0_to_12WithSW_deriv_DAOD_JETM6_test_bin_count.txt",
                "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub05gev_pure_matched/mc16_13TeV_364702_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0_to_12WithSW_deriv_DAOD_JETM6_train_bin_count.txt"]


save_path = "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_esub05gev_pure_matched/"     # target for save
os.makedirs(save_path, exist_ok=True)

if apply_limitor:
    limitor_list = list()
    for item in file_limitor:
        limiting_array = np.loadtxt(item, delimiter=',', dtype='int')
        limitor_list.append(limiting_array)
        #print("Shape of array is: {} and item 10 is {}".format(limiting_array.shape, limiting_array[0, 11]))

for l in range(len(file_path_list)):
    count_energy_per_bin = np.zeros(4001).astype('int') # everything above or equal 4000 GeV is seen as a single bin
    bin = np.arange(4001)
    if apply_limitor:
        limiting_array = limitor_list[l][0,:]   # no need to take bin values, they are the same here
    file_path_sub_list = file_path_list[l]
    file_name = file_name_list[l]

    # make the histogram of energy and cut based on label
    new_list = list()
    energy_list = list()
    counter = 0
    start_counter = 0
    for file_path in file_path_sub_list:
        print(file_path)
        start_counter = counter
        with open(file_path) as json_file:
            data_array = json.load(json_file)['JuniprJets']

        for item in data_array:
            label = item["label"]
            if label == target_label:
                seed_momentum = item["seed_momentum"]
                energy = seed_momentum[0]
                energy_round = np.floor(energy).astype(int)
                if energy_round > 4000:
                    energy_round = 4000
            
                if apply_limitor:
                    if count_energy_per_bin[energy_round] < limiting_array[energy_round]:
                        counter += 1
                        new_list.append(item)
                        energy_list.append(energy)
                        count_energy_per_bin[energy_round] +=1
                else:
                    counter += 1
                    new_list.append(item)
                    energy_list.append(energy)
                    count_energy_per_bin[energy_round] +=1
        print("In total, {} jets were found with the label {} and respecting the energy distribution from {}".format(counter - start_counter, target_label, file_path))
    print("There are {} jets in total for {}".format(counter, file_path))
    new_json = {'JuniprJets':  new_list}
    with open(os.path.join(save_path, file_name + '.json'), "w") as write_file:
       json.dump(new_json, write_file)

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

print("###########################################")
      

