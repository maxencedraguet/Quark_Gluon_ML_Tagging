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
pd.set_option('display.max_rows', None)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from Utils import get_dictionary_cross_section

class DataLoader_Set2(_BaseDataLoader):
    def __init__(self, config: Dict) -> None:
        self.extract_parameters(config)
    
    def extract_parameters(self, config: Dict):
        self.data_path = config.get(["absolute_data_path"])
        self.seed = config.get(["seed"])
        self.fraction_data = config.get(["fraction_data"])
        self.equilibrate_bool = config.get(["equilibrate_data"])
        self.test_size = config.get(["test_size"])

        self.cross_section_based_bool = config.get(["cross_section_sampler", "cs_based"])
        self.n_samples = config.get(["cross_section_sampler", "n_samples"])

    def load_separate_data(self):
        """
        Opens the h5 file, takes the data, and return its separation in train/test.
        """
        print("Start reading data")
        input_file = os.path.join(self.data_path, 'mc16aprocessed.h5')
        self.store = pd.HDFStore(input_file, "r")
        self.store_keys = self.store.keys()
        data_input = pd.DataFrame()
        
        if self.cross_section_based_bool:
            self.rel_cross_section = self.set_up_cross_section_sampler()
            data_input = self.cross_section_sampler(self.store)
        else:
            count = 0
            for key in self.store_keys:
                count += 1
                print('Count: {0}, key: {1}'.format(count, key))
                data_input = data_input.append(self.store[key].sample(frac = self.fraction_data, random_state = self.seed))
                    #if (count == 2):
                    #break
                    
        self.store.close()
        print("Initial state")
        self.analyse_dataset(data_input[['isTruthQuark']])
        if self.equilibrate_bool:
            data_input = self.equilibrate(data_input)
        print("\nEquilibrated state")
        self.analyse_dataset(data_input[['isTruthQuark']])
        print(data_input)
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

    def set_up_cross_section_sampler(self):
        """
        This gathers a dictionnary of cross section and limits it to the sample available.
        """
        whole_dictionary = get_dictionary_cross_section()
        local_dictionary  = dict()
        total_value = 0
        for key in self.store_keys:
            dsid = key.split(".")[4]
            cross_sec, k_fac = whole_dictionary[dsid]
            local_value = float(cross_sec) * float(k_fac)
            local_dictionary[key] = local_value
            total_value += local_value
    
        for key in self.store_keys:
            local_dictionary[key] = local_dictionary[key] / total_value

        return local_dictionary

    def cross_section_sampler(self, store_file):
        """
        Specific kind of sampler. Is based on cross section. With too rare events furnishing a single sample.
        """
        data = pd.DataFrame()
        count = 0
        diff = 0
        sum = 0
        for key in self.store_keys:
            count += 1
            if count == 5 :
                break
            #print('Count: {0}, key: {1} and relative cross section {2}'.format(count, key, self.rel_cross_section[key]))
            analysed_file = store_file.get_storer(key)
            number_entries = analysed_file.shape[0]
            number_samples = np.ceil(self.n_samples * self.rel_cross_section[key])
            number_samples_expected = number_samples
            if number_entries == 1:
                print('Loading empty file {}, expecting {}'.format(key, number_samples_expected))
                continue
            if number_samples < 1:
                number_samples = 1
            else:
                number_samples = int(number_samples)
            
            if number_samples > number_entries:
                diff += number_samples - number_entries
                print('Asking too many samples out of file {}: demanding {} but having {}'.format(key, number_samples, number_entries))
                number_samples = number_entries
            store_file.get_storer(key).shape[0]
            data = data.append(store_file[key].sample(n = number_samples, random_state = self.seed))
            sum += number_samples
            print("From file {} taking {}, expected {}\n".format(key, number_samples, number_samples_expected))
                
        print("Finished cross-section sampling. Events sampled {} | events demanded {}".format(sum, self.n_samples) )
        return data
