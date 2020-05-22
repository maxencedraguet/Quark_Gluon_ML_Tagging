#############################################################################
#
# DataLoader_Set1.py
#
# A data loader for the first set of data: Set1. This one requires a CSV.
#
# Data from: /data/atlas/atlasdata3/xAOD/MultiJets/NTuples/20200506_qgTaggingSystematics/
# Data produced by: Aaron O'Neill.
#
# Author -- Maxence Draguet (19/05/2020)
#
# Entries are
#
# entry, subentry, jetPt, jetEta, jetPhi, jetMass, jetEnergy, jetEMFrac, jetHECFrac, jetChFrac,
# jetNumTrkPt500, jetNumTrkPt1000, jetTrackWidthPt500, jetTrackWidthPt1000, jetSumTrkPt500,
# jetSumTrkPt1000, partonIDs, BDTScore, isTruthQuark, isBDTQuark, isnTrkQuark
#
#############################################################################
import os
import sys
from typing import Dict

from .BaseDataLoader import _BaseDataLoader

import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader_Set1(_BaseDataLoader):
    def __init__(self, config: Dict) -> None:
        self.extract_parameters()

    def extract_parameters(self, config: Dict):
        self.data_path = config.get(["relative_data_path"])
        self.seed = config.get(["seed"])
        self.test_size = config.get(["BDT_model", "test_size"])
    
    def load_separate_data(self):
        """
        Opens the csv file, takes the data, and return its separation in train/test.
        """
        print("Start Reading Data")
        input_file = self.data_path + "user.aoneill.21148352.NTUP._000050.root_all.csv"
        data_input = np.loadtxt(fname = input_file, delimiter=',', skiprows=1, max_rows = 100)# MODIFY max_rows = 100000
        data_output = data_input[:, 18]
        input_train, input_test, output_train, output_test = train_test_split(data_input,
                                                                              data_output,
                                                                              test_size = self.test_size)
        print("Train")
        self.analyse_dataset(output_train)
        print("Test")
        self.analyse_dataset(output_test)
        # Limit to training information
        input_train = input_train[:,2:15]
        input_test  = input_test[:,2:15]
        
        self.data = {}
        self.data["input_train"] = input_train
        self.data["input_test"] = input_test
        self.data["output_train"] = output_train
        self.data["output_test"] = output_test
        return self.data

    def analyse_dataset(self, data)->None:
        keys, counts = np.unique(data, return_counts=True)
        res_dict = dict(zip(keys, counts))
        for item in res_dict.keys():
            print('For {}: {}'.format(int(item), int(res_dict[item])))
        print('Fraction of {0} is {1:2.2f}'.format(int(keys[0]), res_dict[keys[0]]/sum(res_dict.values())))




