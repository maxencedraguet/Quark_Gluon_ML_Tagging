#############################################################################
#
# DataLoader_Set2.py
#
# A data loader for the second set of data: Set2. This one requires a HDF5 file.
#
# Data from:  /data/atlas/atlasdata3/mdraguet/Set2/HF/
# Data produced by Aaron O'Neill and processed by Maxence Draguet.
#
# Author -- Maxence Draguet (27/05/2020)
#
# This loader takes directly the combined and processed root files  stored in a HDF5 file
# and convert them into the expected format.
#
#['jetPt', 'jetEta', 'jetPhi', 'jetMass', 'jetEnergy', 'jetEMFrac', 'jetHECFrac', 'jetChFrac',
# 'jetNumTrkPt500', 'jetNumTrkPt1000', 'jetTrackWidthPt500', 'jetTrackWidthPt1000', 'jetSumTrkPt500',
# 'jetSumTrkPt1000', 'partonIDs', 'BDTScore', 'isTruthQuark', 'isBDTQuark', 'isnTrkQuark']
#
#############################################################################
import os
import sys
from typing import Dict

from .BaseDataLoader import _BaseDataLoader

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

class DataLoader_Set2(_BaseDataLoader):
    def __init__(self, config: Dict) -> None:
        self.extract_parameters(config)
    
    def extract_parameters(self, config: Dict):
        self.data_path = config.get(["absolute_data_path"])
        self.seed = config.get(["seed"])
        self.fraction_data = config.get(["fraction_data"])
        self.equilibrate_bool = config.get(["equilibrate_data"])
        
        self.test_size = config.get(["BDT_model", "test_size"])
 
    def load_separate_data(self):
        """
        Opens the h5 file, takes the data, and return its separation in train/test.
        """
        print("Start reading data")
        input_file = os.path.join(self.data_path, 'mc16aprocessed.h5')
        self.store = pd.HDFStore(input_file, "r")
        self.store_keys = self.store.keys()
        data_input = pd.DataFrame()
        count = 0
        for key in self.store_keys:
            count += 1
            print('Count: {0}, key: {1}'.format(count, key))
            data_input = data_input.append(self.store[key].sample(frac = self.fraction_data, random_state = self.seed))
                #if (count == 10):
                #break
        self.store.close()
        print("Initial state")
        self.analyse_dataset(data_input[['isTruthQuark']])
        if self.equilibrate_bool:
            data_input = self.equilibrate(data_input)
        print("\nEquilibrated state")
        self.analyse_dataset(data_input[['isTruthQuark']])
        data_output = data_input[['isTruthQuark']]
        data_output_BDT = data_input[['BDTScore']]* (-1) #Problem with the BDT values of Baltz: multiply by -1
        # Scale the inputs and restrict to training variables. Note that this outputs a numpy array
        data_input = scale(data_input.loc[:,'jetPt':'jetSumTrkPt1000']) #
        #print(pd.DataFrame.from_records(data_input).describe())
        input_train, input_test, output_train, output_test, BDT_output_train, BDT_output_test = train_test_split(data_input,
                                                                              data_output,
                                                                              data_output_BDT,
                                                                              test_size = self.test_size,
                                                                              random_state = self.seed)
        print("\nTraining set")
        self.analyse_dataset(output_train)
        print("\nTesting set")
        self.analyse_dataset(output_test)

        output_train = output_train.values.ravel()
        output_test = output_test.values.ravel()
        
        data = {}
        data["input_train"] = input_train
        data["input_test"] = input_test
        data["output_train"] = output_train
        data["output_test"] = output_test
        data["output_BDT_train"] = BDT_output_train
        data["output_BDT_test"] = BDT_output_test
        return data

    def analyse_dataset(self, data)->None:
        """
        Return the fraction of true quark jet in the dataset
        """
        reduced = data['isTruthQuark'].value_counts()
        print("Number of quark = ", reduced[1])
        print("Fraction of quark is: ", reduced[1]/(reduced[1] + reduced[0]))

    def equilibrate(self, data):
        """
        Equilibrate the dataset to have the same number of occurence of 'isTruthQuark' for each class
        """
        number_q = data[data['isTruthQuark'] ==1]['isTruthQuark'].count()
        number_g = data[data['isTruthQuark'] ==0]['isTruthQuark'].count()
        minimum = min(number_g, number_q)

        data_q = data[data['isTruthQuark'] ==1].sample(n = minimum, random_state = self.seed)
        data_g = data[data['isTruthQuark'] ==0].sample(n = minimum, random_state = self.seed)
        return data_g.append(data_q)

