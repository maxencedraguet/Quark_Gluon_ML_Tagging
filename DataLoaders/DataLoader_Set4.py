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
import multiprocessing

from .BaseDataLoader import _BaseDataLoader

import numpy as np
import pandas as pd
#pd.set_option('display.max_rows', None)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data import random_split

from Utils import get_dictionary_cross_section
from .JuniprDataset import JuniprDataset, PadTensors, PadTensorsWithMask, FeatureScalingOwn, FeatureScalingJunipr, GranulariseBranchings, AddExtraLabel, CorrectTrueLabel
from .JuniprLargeDataset import JuniprLargeDataset

class DataLoader_Set4(_BaseDataLoader):
    def __init__(self, config: Dict) -> None:
        self.extract_parameters(config)

    def extract_parameters(self, config: Dict):
        self.data_path = config.get(["absolute_data_path"]) # In this case, the path the full path to a specific json file or a txt file with the chunks.
        self.chunked_dataset_bool = config.get(["chunked_dataset"])
        self.seed = config.get(["seed"])
        self.fraction_data = config.get(["fraction_data"])
        self.test_size = config.get(["test_size"])
        
        self.cross_section_based_bool = config.get(["cross_section_sampler", "cs_based"])
        self.n_samples = config.get(["cross_section_sampler", "n_samples"])
        
        self.train_bool  = config.get(["Junipr_Model", "train"])   # If true, the return will contain training and validations set and keep them in the class
        self.assess_bool = config.get(["Junipr_Model", "assess"])  # If true, the return will contain return a test set and keep it in the class
        
        self.num_workers = config.get(["Junipr_Model", "num_workers"])
        if self.chunked_dataset_bool:
            self.num_workers = 0
        
        self.binary_junipr = config.get(["Junipr_Model", "binary_runner_bool"]) # whether the data is for a binary junipr or not
        if self.binary_junipr:
            # In this case, the path the full path to be completed by "_train.json" and "_test.json".
            self.data_path_gluon = config.get(["Junipr_Model", "binary_runner", "gluon_data_path" ])
            self.data_path_quark = config.get(["Junipr_Model", "binary_runner", "quark_data_path" ])
        self.padding_size = config.get(["Junipr_Model", "Junipr_Dataset", "padding_size"])
        self.padding_value= config.get(["Junipr_Model", "Junipr_Dataset", "padding_value"])
        self.feature_scaling_params = config.get(["Junipr_Model", "Junipr_Dataset", "feature_scaling_parameters"])
        self.granularity = config.get(["Junipr_Model", "Junipr_Dataset", "granularity"])
        self.validation_size= config.get(["Junipr_Model", "Junipr_Dataset", "validation_size"])

    def load_separate_data(self):
        """
        Opens the json file, set up a JuniprDataset with necessary tranforms, and returns datasets required for the JUNIPR program (training/validating and/or testing).
        """
        if self.binary_junipr:
            print("Start reading data for binary Junipr")
            # Set up the transforms to apply to the dataset
            train_pading = PadTensorsWithMask(self.padding_size, self.padding_value, train_bool = True)
            #train_scaling = FeatureScalingJunipr(self.feature_scaling_params, train_bool = True)
            train_scaling = FeatureScalingOwn(train_bool = True)
            train_granularise = GranulariseBranchings(self.granularity, train_bool = True)
            train_add_gluon_set = CorrectTrueLabel(0, train_bool = True)
            train_add_quark_set = CorrectTrueLabel(1, train_bool = True)
            
            test_pading = PadTensorsWithMask(self.padding_size, self.padding_value, train_bool = False)
            #test_scaling = FeatureScalingJunipr(self.feature_scaling_params, train_bool = False)
            test_scaling = FeatureScalingOwn(train_bool = False)
            self.used_test_scaling = test_scaling
            test_granularise = GranulariseBranchings(self.granularity, train_bool = False)
            test_add_gluon_set = CorrectTrueLabel(0, train_bool = False)
            test_add_quark_set = CorrectTrueLabel(1, train_bool = False)
            # Compose these, beware of the order.
            train_composed_quark = transforms.Compose([train_scaling, train_granularise, train_pading, train_add_quark_set])
            train_composed_gluon = transforms.Compose([train_scaling, train_granularise, train_pading, train_add_gluon_set])
            test_composed_quark = transforms.Compose([test_scaling, test_granularise, test_pading, test_add_quark_set])
            test_composed_gluon = transforms.Compose([test_scaling, test_granularise, test_pading, test_add_gluon_set])
            #composed = transforms.Compose([scaling, onehot])
            # Create a JuniprDataset class (inheriting from the PyTorch dataset class) with the data contained in self.data_path
            
            if self.chunked_dataset_bool:
                # Only train data is chunked this case. The test data can be defined as usual (and the validation data too). If need to generalise, a chunk will be used for validation (chunks are randomised).
                # The chunks setting needs the dataset to be balanced.
                with open(os.path.join(self.data_path_gluon, 'chunk_train_datasets.txt')) as train_file_list:
                    gluon_chunk_path_train = [(int(i.split(',')[0]), i.split(',')[1].split('\n')[0]) for i in train_file_list]
                with open(os.path.join(self.data_path_quark, 'chunk_train_datasets.txt')) as train_file_list:
                    quark_chunk_path_train = [(int(i.split(',')[0]), i.split(',')[1].split('\n')[0]) for i in train_file_list]
                
                with open(os.path.join(self.data_path_gluon, 'chunk_test_datasets.txt')) as test_file_list:
                    gluon_chunk_path_test = [(int(i.split(',')[0]), i.split(',')[1].split('\n')[0]) for i in test_file_list]
                with open(os.path.join(self.data_path_quark, 'chunk_test_datasets.txt')) as test_file_list:
                    quark_chunk_path_test = [(int(i.split(',')[0]), i.split(',')[1].split('\n')[0]) for i in test_file_list]
                
                test_data_path_quark = quark_chunk_path_test[0][1]
                test_data_path_gluon = gluon_chunk_path_test[0][1]
                
                # This already contains both datasets and is randomised on chunks (not across chunks however).
                train_set = JuniprLargeDataset([quark_chunk_path_train, gluon_chunk_path_train], self.seed, train_bool = True, transform = train_composed_quark, binary_junipr = True)
                
                gluon_dataset_test  = JuniprDataset(test_data_path_quark, train_bool = False, transform = test_composed_gluon)
                quark_dataset_test  = JuniprDataset(test_data_path_gluon, train_bool = False, transform = test_composed_quark)
                
            else:
                if self.num_workers != 0:
                    gluon_manager_train = multiprocessing.Manager()
                    quark_manager_train = multiprocessing.Manager()
                    gluon_manager_test  = multiprocessing.Manager()
                    quark_manager_test  = multiprocessing.Manager()
                else:
                    gluon_manager_train = None
                    quark_manager_train = None
                    gluon_manager_test  = None
                    quark_manager_test  = None

                gluon_dataset_train = JuniprDataset(self.data_path_gluon+"_train.json", train_bool = True, transform = train_composed_gluon, manage_with =gluon_manager_train)
                quark_dataset_train = JuniprDataset(self.data_path_quark+"_train.json", train_bool = True, transform = train_composed_quark, manage_with =quark_manager_train)
                
                gluon_dataset_test  = JuniprDataset(self.data_path_gluon+"_test.json", train_bool = False, transform = test_composed_gluon, manage_with =gluon_manager_test)
                quark_dataset_test  = JuniprDataset(self.data_path_quark+"_test.json", train_bool = False, transform = test_composed_quark, manage_with =quark_manager_test)

                # Force the datasets to have as many gluons as quarks for training
                need_train_matching_bool = False
                if len(gluon_dataset_train) != len(quark_dataset_train):
                    need_train_matching_bool = True
                
                need_test_matching_bool  = False
                if len(gluon_dataset_train) != len(quark_dataset_train):
                    need_test_matching_bool = True

                print("Does binary JUNIPR needs a matching of train datasets? {}. And of test datasets? {}.".format(need_train_matching_bool, need_test_matching_bool))

                if need_train_matching_bool:
                    min_train_size = int(len(gluon_dataset_train))
                    gluon_split_train = True    # gluon is the smallest one
                    if min_train_size > int(len(quark_dataset_train)):
                        min_train_size = int(len(quark_dataset_train))
                        gluon_split_train  = False # quark is the smallest one
                    if gluon_split_train:   # gluon is smallest, split quark
                        quark_dataset_train, _ = random_split(quark_dataset_train, [min_train_size, len(quark_dataset_train) - min_train_size])
                    else:                   # gluon is largest, split it
                        gluon_dataset_train, _ = random_split(gluon_dataset_train, [min_train_size, len(gluon_dataset_train) - min_train_size])

                if need_test_matching_bool:
                    min_test_size = int(len(gluon_dataset_test))
                    gluon_split_test = True
                    if min_test_size > int(len(quark_dataset_test)):
                        min_test_size = int(len(quark_dataset_test))
                        gluon_split_test = False
                    if gluon_split_test:   # gluon is smallest, split quark
                        quark_dataset_test, _ = random_split(quark_dataset_test, [min_test_size, len(quark_dataset_test) - min_test_size])
                    else:                   # gluon is largest, split it
                        gluon_dataset_test, _ = random_split(gluon_dataset_test, [min_test_size, len(gluon_dataset_test) - min_test_size])

                if need_train_matching_bool or need_test_matching_bool:
                    print("The smallest dataset sizes were {} for train (due to gluon ? {}) and {} for test (due to gluon ? {})".format(min_train_size, gluon_split_train, min_test_size, gluon_split_test))
                    print("Checking: are the training datasets now equilibrated ? {}. And the testing ones ? {}.".format(len(quark_dataset_train) ==  len(gluon_dataset_train), len(quark_dataset_test) ==  len(gluon_dataset_test)))
                train_set = torch.utils.data.ConcatDataset([quark_dataset_train, gluon_dataset_train])
                #Remark: it is imperative to shuffle them for the training dataloader (otherwise you'll first read all quarks and then all gluons)

            min_test_size = int(len(gluon_dataset_test))
            # Cutting the test sets into validations and true tests
            val_size = int(min_test_size * self.validation_size)
            test_size = min_test_size - val_size
            quark_val_set, quark_test_set = random_split(quark_dataset_test, [val_size, test_size])
            gluon_val_set, gluon_test_set = random_split(gluon_dataset_test, [val_size, test_size])

            # Concatenate these test and validations datasets now.Might need to shuffle the test set when assessing (otherwise you'll only plot the x first jets which will be quarks).
            val_set   = torch.utils.data.ConcatDataset([quark_val_set, gluon_val_set])
            test_set  = torch.utils.data.ConcatDataset([quark_test_set, gluon_test_set])
        else:
            print("Start reading data for unary Junipr")
            # Set up the transforms to apply to the dataset
            #composed = transforms.Compose([scaling, onehot])
            
            #train_pading = PadTensors(self.padding_size, self.padding_value, train_bool = True)
            train_pading = PadTensorsWithMask(self.padding_size, self.padding_value, train_bool = True)
            train_scaling = FeatureScalingOwn(train_bool = True)
            train_granularise = GranulariseBranchings(self.granularity, train_bool = True)
            test_pading = PadTensorsWithMask(self.padding_size, self.padding_value, train_bool = False)
            test_scaling = FeatureScalingOwn(train_bool = False)
            self.used_test_scaling = test_scaling
            test_granularise = GranulariseBranchings(self.granularity, train_bool = False)

            # Compose these, beware of the order.
            train_composed = transforms.Compose([train_scaling, train_granularise, train_pading])
            test_composed  = transforms.Compose([test_scaling, test_granularise, test_pading])
            # Create a JuniprDataset class (inheriting from the PyTorch dataset class) with the data contained in self.data_path

            if self.chunked_dataset_bool:
                with open(os.path.join(self.data_path, 'chunk_train_datasets.txt')) as train_file_list:
                    chunk_path_train = [(int(i.split(',')[0]), i.split(',')[1].split('\n')[0]) for i in train_file_list]
                with open(os.path.join(self.data_path, 'chunk_test_datasets.txt')) as test_file_list:
                    chunk_path_test = [(int(i.split(',')[0]), i.split(',')[1].split('\n')[0]) for i in test_file_list]
                dataset_train = JuniprLargeDataset(chunk_path_train, self.seed, train_bool = True, transform = train_composed)
                dataset_test = JuniprLargeDataset(chunk_path_test, self.seed, train_bool = False, transform = test_composed)
            else:
                if self.num_workers != 0:
                    manager_train = multiprocessing.Manager()
                    manager_test = multiprocessing.Manager()
                else:
                    manager_train = None
                    manager_test  = None
                dataset_train = JuniprDataset(self.data_path+"_train.json", train_bool = True, transform = train_composed, manage_with = manager_train)
                dataset_test = JuniprDataset(self.data_path+"_test.json", train_bool = False, transform = test_composed, manage_with = manager_test)
            
            test_dataset_size_used = int(len(dataset_test))
            val_size = int(test_dataset_size_used * self.validation_size)
            test_size = test_dataset_size_used - val_size
            val_set, test_set = random_split(dataset_test, [val_size, test_size])
            
            train_set, val_set, test_set = dataset_train, val_set, test_set
        
        # place in memory what is needed:
        if self.train_bool:
            self.train_set = train_set
            self.val_set = val_set
            print("Side of train: ", len(self.train_set))
            print("Side of validation: ", len(self.val_set))
        if self.assess_bool:
            self.test_set = test_set
            print("Side of test: ", len(self.test_set))

        if self.train_bool and not(self.assess_bool):
            # Need train but not test
            return self.train_set, self.val_set, 0
        elif self.train_bool and self.assess_bool:
            # Need train and test
            return self.train_set, self.val_set, self.test_set

        elif self.assess_bool:
            # Only needs test
            return 0, 0, self.test_set
        else:
            # impossible: need either train nor test, cannot require nothing (nothing to do):
            raise ValueError("No task submitted to JUNIPR. Train: {}. Test: {}". format(self.train_bool, self.assess_bool))








