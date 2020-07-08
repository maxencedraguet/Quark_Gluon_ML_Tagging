#############################################################################
#
# DataLoader_Set4.py
#
# A data loader for the 4th and final set of data. Reads a json files into a tensor compatible with PyTorch
#
# Data from:  /data/atlas/atlasdata3/mdraguet/Set4/junipr/
# Data produced by Maxence Draguet (from GranularGatherer and processed by GranularTransformar).
#
# Author -- Maxence Draguet (06/07/2020)
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

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data import random_split

from Utils import get_dictionary_cross_section
from .JuniprDataset import JuniprDataset, PadTensors, FeatureScaling, OneHotBranch

class DataLoader_Set4(_BaseDataLoader):
    def __init__(self, config: Dict) -> None:
        self.extract_parameters(config)

    def extract_parameters(self, config: Dict):
        self.data_path = config.get(["absolute_data_path"]) # In this case, the path the full path to a specific json file.
        self.seed = config.get(["seed"])
        self.fraction_data = config.get(["fraction_data"])
        #self.equilibrate_bool = config.get(["equilibrate_data"]) # not in used here
        self.test_size = config.get(["test_size"])
        
        self.cross_section_based_bool = config.get(["cross_section_sampler", "cs_based"])
        self.n_samples = config.get(["cross_section_sampler", "n_samples"])

        self.padding_size = config.get(["Junipr_Model", "Junipr_Dataset", "padding_size"])
        self.padding_value= config.get(["Junipr_Model", "Junipr_Dataset", "padding_value"])
        self.feature_scaling_params = config.get(["Junipr_Model", "Junipr_Dataset", "feature_scaling_parameters"])
        self.granularity = config.get(["Junipr_Model", "Junipr_Dataset", "granularity"])
        #self.padding_size = config.get(["JuniprDataset", "padding_size"])

    def load_separate_data(self):
        """
        Opens the json file, set up a JuniprDataset with necessary tranforms, and returns two datasets (for training and testing).
        """
        print("Start reading data")
        # Set up the transforms to apply to the dataset
        pading = PadTensors(self.padding_size, self.padding_value)
        scaling = FeatureScaling(self.feature_scaling_params)
        onehot = OneHotBranch(self.granularity)
        # Compose these, beware of the order.
        composed = transforms.Compose([scaling, onehot, pading])
        
        # Create a JuniprDataset class (inheriting from the PyTorch dataset class) with the data contained in self.data_path
        dataset = JuniprDataset(self.data_path, transform = composed)
        
        # Cut this dataset into two parts: train and test (note that a seed for PyTorch is set in the main so the randomness of the operation is controlled)
        dataset_size_used = int(len(dataset) * self.fraction_data)
        test_size = int(dataset_size_used * self.test_size)
        train_size = dataset_size_used - test_size
        self.train_set, self.test_set = random_split(dataset, [train_size, test_size])
        
        """
        # Instantiate two dataloaders: for train and test
        self.train_dataloader = DataLoader(train_set, batch_size=2, shuffle=True)
        self.test_dataloader  = DataLoader(test_set,  batch_size=2, shuffle=True)
        """
        return self.train_set, self.test_set







