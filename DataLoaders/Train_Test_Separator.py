#############################################################################
#
# Train_Test_Separator.py
#
# A short codebase to separate the global h5 files into
# a train-validation-test set of files.
#
#############################################################################

import os
import sys
import warnings
import tables
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd         #pd.set_option('display.max_rows', None)
import fnmatch
import pprint
from sklearn.model_selection import train_test_split

class Train_Test_Separator(ABC):
    def __init__(self, config: Dict):
        self.extract_parameters(config)
        self.load_separate()
    
    def extract_parameters(self, config: Dict)->None:
        # In this case it should be the path to the h5 file itself.
        self.data_file = config.get(["absolute_data_path"])
        self.save_path = config.get(["Train_Test_Separator", "save_path"])
        os.makedirs(self.save_path, exist_ok=True)
        self.test_size = config.get(["Train_Test_Separator", "test_size"])
        
        # fraction of the global dataset for validation
        whole_validation_size = config.get(["Train_Test_Separator", "validation_size"])
        # fraction of the train set that needs to be cut for the valdiation set
        self.validation_size = whole_validation_size / (1.0 - self.test_size)
        self.seed = config.get(["seed"])

    def load_separate(self):
        """
        Creates train/validation/test h5 files with keys being file names.
        """
        self.store = pd.HDFStore(self.data_file, "r")
        self.store_keys = self.store.keys()
        count = 0
        bug_keys = []
        too_small_keys = []
        original_warnings = list(warnings.filters)
        warnings.simplefilter('ignore', tables.NaturalNameWarning)
        for key in self.store_keys:
            count += 1
            print('Count: {0}, key: {1}'.format(count, key))
            data_input = pd.DataFrame()
            data_input = data_input.append(self.store[key])
            if data_input.shape[0] == 0:
                print("Problem, empty dataset!")
                bug_keys.append(key)
                continue
            if data_input.shape[0] <5:
                print("Problem, less than 5 elements!")
                too_small_keys.append(key)
                continue
            train, test = train_test_split(data_input,
                                           test_size = self.test_size,
                                           random_state = self.seed)
            train, validation = train_test_split(train,
                                                 test_size = self.validation_size,
                                                 random_state = self.seed)
        
            train.to_hdf(os.path.join(self.save_path, 'train.h5'), key = key)
            validation.to_hdf(os.path.join(self.save_path, 'validation.h5'), key = key)
            test.to_hdf(os.path.join(self.save_path, 'test.h5'), key = key)
                #if count > 3:
                #break
        warnings.filters = original_warnings
        self.store.close()
        with open(os.path.join(self.save_path, 'empty_files.txt'), 'w') as f:
            for elem in bug_keys:
                f.write("%s\n" % str(elem))
        with open(os.path.join(self.save_path, 'less_than_5_elem_files.txt'), 'w') as f:
            for elem in too_small_keys:
                f.write("%s\n" % str(elem))
