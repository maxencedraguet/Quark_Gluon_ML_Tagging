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
# This loader takes directly the root files and convert them into the expected
# format using UpRoot. It does not limit itself to the "...all.csv".
#
#############################################################################
import os
import sys
from typing import Dict

from .BaseDataLoader import _BaseDataLoader

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader_Set2(_BaseDataLoader):
    def __init__(self, config: Dict) -> None:
        self.data_path = config.get(["relative_data_path"])
        self.seed = config.get(["seed"])
        self.test_size = config.get(["BDT_model", "test_size"])

    def load_separate_data(self):
        """
        Opens the csv file, takes the data, and return its separation in train/test.
        """
        print("Start reading data")
        input_file = self.data_path + "user.aoneill.21148352.NTUP._000050.root_all.csv"
        data_input = np.loadtxt(fname = input_file, delimiter=',', skiprows=1, max_rows = 10000)# MODIFY
        data_output = data_input[:, 18]
        input_train, input_test, output_train, output_test = train_test_split(data_input,
                                                                              data_output,
                                                                              test_size = self.test_size)
        # Limit to training information
        input_train = input_train[:,2:15]
        input_test  = input_test[:,2:15]
        
        data = {}
        data["input_train"] = input_train
        data["input_test"] = input_test
        data["output_train"] = output_train
        data["output_test"] = output_test
        #print(output_test)
        #print(input_train)
        return data
    
    def list_input(self):
        pass

