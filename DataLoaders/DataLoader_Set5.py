#############################################################################
#
# DataLoader_Set5.py
#
# A data loader for the NN/BDT with the jet table reconstructed from DAOD_JETM6
#
# Data from:  /data/atlas/atlasdata3/mdraguet/Set4_dijet_ttbar_esub1gev_matched_Completely_H5
#      Uses the follwing files in the folder: gluon.h5  quark.h5
# Data produced and processed by Maxence Draguet.
#
# Author -- Maxence Draguet (18/08/2020)
#
# This loader takes directly the combined and processed root files  stored in a HDF5 file
# and convert them into the expected format.
#
# These are the available variables:
#
# jetPt, jetEta, jetPhi, jetMass, jetE, jetWidth, jetEMFrac, jetChFrac, jetNumTrkPt500, jetNumTrkPt1000, jetTrackWidthPt1000, jetSumTrkPt500, partonID, isTruthQuark, jetNumberConstituent, isNotPVJet
#
# Problem with jetTrackWidthPt1000, it's a list. It has to be dropped here
#
#############################################################################
import os
import sys
from typing import Dict

from .BaseDataLoader import _BaseDataLoader

import numpy as np
import pandas as pd
#pd.set_option('display.max_rows', None)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from Utils import get_dictionary_cross_section

class DataLoader_Set5(_BaseDataLoader):
    def __init__(self, config: Dict) -> None:
        self.extract_parameters(config)
    
    def extract_parameters(self, config: Dict):
        self.data_path = config.get(["absolute_data_path"])
        self.seed = config.get(["seed"])
        self.fraction_data = config.get(["fraction_data"])
        self.test_size = config.get(["test_size"])
        self.list_variables = ['jetPt', 'jetEta', 'jetPhi', 'jetMass', 'jetE', 'jetWidth', 'jetEMFrac','jetChFrac', 'jetNumTrkPt500', 'jetNumTrkPt1000', 'jetSumTrkPt500']

    def load_separate_data(self):
        """
        Opens the h5 file, takes the data, and return its separation in train/test.
        """
        print("Start reading data")
        quark_input_file = os.path.join(self.data_path, 'quark.h5')
        gluon_input_file = os.path.join(self.data_path, 'gluon.h5')
        
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
            quark_data_input = quark_data_input.append(quark_file[key].sample(frac = self.fraction_data, random_state = self.seed))

        count = 0
        for key in store_gluon_keys:
            count += 1
            print('Count: {0}, key: {1}'.format(count, key))
            gluon_data_input = gluon_data_input.append(gluon_file[key].sample(frac = self.fraction_data, random_state = self.seed))

        quark_file.close()
        gluon_file.close()

        # Combine the two frames
        data_input = quark_data_input.append(gluon_data_input, ignore_index=True)
        data_output = data_input[['isTruthQuark']]

        # Remove un-usable variables. Note that the energy is still in though it's exactly the same for both.
        data_input = data_input.loc[:,'jetPt':'jetSumTrkPt500']
        data_input.drop(['jetTrackWidthPt1000'], axis = 1, inplace=True) # problem with jetTrackWidthPt1000, it's a list!, drop it.
        print("Variables used: \n", data_input.columns)
        #print("data_input types, ", data_input.dtypes)
        
        # Scale the inputs and restrict to training variables. Note that this outputs a numpy array
        data_input = scale(data_input)
        # Separate a train and test set randomly.
        input_train, input_test, output_train, output_test = train_test_split(data_input,
                                                                              data_output,
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
        data["output_BDT_train"] = None
        data["output_BDT_test"] = None
        return data

    def analyse_dataset(self, data)->None:
        """
        Return the fraction of true quark jet in the dataset
        """
        reduced = data['isTruthQuark'].value_counts()
        print("Number of quark = ", reduced[1])
        print("Fraction of quark is: ", reduced[1]/(reduced[1] + reduced[0]))

