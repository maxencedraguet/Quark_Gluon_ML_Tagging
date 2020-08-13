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
#pd.set_option('display.max_rows', None)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data import random_split

from Utils import get_dictionary_cross_section
from .JuniprDataset import JuniprDataset, PadTensors, PadTensorsWithMask, FeatureScalingOwn, FeatureScalingJunipr, GranulariseBranchings, AddExtraLabel, CorrectTrueLabel

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

    def undo_mapping_FeatureScalingOwn(self, processed_dictionnary_input, processed_dictionnary_output):
        """
        Takes as input a dictionnary of inputs and a dictionnary of outputs (of a junipr model) that have been processed by a FeatureScalingOwn.
        By virtue of the pytorch implementation, the activation functions have not yet been applied to the output dictionnary (since they are contained in the loss module for numerical stability). This is done here.
        Returns two dictionnaries (for input and output of JUNIPR) withtout scaling in the daughter branch.
        """
        pass
        """
        dictionnary_input  = dict()
        dictionnary_output = dict()
        
        output_end      = processed_dictionnary_output["output_end"]
        output_very_end = processed_dictionnary_output["output_very_end"]
        output_mother   = processed_dictionnary_output["output_mother"]
        output_branch_z = processed_dictionnary_output["output_branch_z"]
        output_branch_t = processed_dictionnary_output["output_branch_t"]
        output_branch_p = processed_dictionnary_output["output_branch_p"]
        output_branch_d = processed_dictionnary_output["output_branch_d"]

        output_end = torch.cat([output_end, output_very_end], dim = 1)
        
        output_end      = torch.sigmoid(output_end)
        output_mother   = nn.Softmax(output_mother)
        output_branch_z = nn.Softmax(output_branch_z)
        output_branch_t = nn.Softmax(output_branch_t)
        output_branch_p = nn.Softmax(output_branch_p)
        output_branch_d = nn.Softmax(output_branch_d)
        
        recurrent_limit = unshaped_output_mother.size()[1]
        
        input_n_branchings = processed_dictionnary_input["n_branchings"]
        input_branching    = processed_dictionnary_input["branching"][:, :recurrent_limit, :]
        input_mother_id    = processed_dictionnary_input["mother_id_energy"][:, :recurrent_limit]
        input_branch_mask  = processed_dictionnary_input["branching_mask"][:, :recurrent_limit]
        
        input_branch_z     = input_branching[:, :, 0]
        input_branch_t     = input_branching[:, :, 1]
        input_branch_p     = input_branching[:, :, 2]
        input_branch_d     = input_branching[:, :, 3]
        

        def unscaler_function(variable, limit, parameter):
            #The opposite of the scaler_function
            return (np.exp(variable * np.log(1 + parameter * limit) ) - 1) / parameter
    
        input_unscaled_z     = unscaler_function(processed_dictionnary_input[], self.limit_z, self.parameter_z)
        input_unscaled_theta = unscaler_function(processed_dictionnary_input[], self.limit_theta, self.parameter_z)
        input_unscaled_delta = unscaler_function(processed_dictionnary_input[], self.limit_delta, self.parameter_z)
        input_unscaled_phi   = processed_dictionnary_input[] * 2 * np.pi
    
        output_unscaled_z     = unscaler_function(processed_dictionnary_output[], self.limit_z, self.parameter_z)
        output_unscaled_theta = unscaler_function(processed_dictionnary_output[], self.limit_theta, self.parameter_theta)
        output_unscaled_delta = unscaler_function(processed_dictionnary_output[], self.limit_delta, self.parameter_delta)
        output_unscaled_phi   = processed_dictionnary_output[] * 2 * np.pi
        """
    
    
    
        # need to
    

    def load_separate_data(self):
        """
        Opens the json file, set up a JuniprDataset with necessary tranforms, and returns two datasets (for training and testing).
        """
        if self.binary_junipr:
            print("Start reading data for binary Junipr")
            # Set up the transforms to apply to the dataset
            train_pading = PadTensorsWithMask(self.padding_size, self.padding_value, train_bool = True)
            #train_scaling = FeatureScalingJunipr(self.feature_scaling_params, train_bool = True)
            train_scaling = FeatureScalingOwn(train_bool = True)
            train_onehot = GranulariseBranchings(self.granularity, train_bool = True)
            train_add_gluon_set = CorrectTrueLabel(0, train_bool = True)
            train_add_quark_set = CorrectTrueLabel(1, train_bool = True)
            
            test_pading = PadTensorsWithMask(self.padding_size, self.padding_value, train_bool = False)
            #test_scaling = FeatureScalingJunipr(self.feature_scaling_params, train_bool = False)
            test_scaling = FeatureScalingOwn(train_bool = False)
            self.used_test_scaling = test_scaling
            test_onehot = GranulariseBranchings(self.granularity, train_bool = False)
            test_add_gluon_set = CorrectTrueLabel(0, train_bool = False)
            test_add_quark_set = CorrectTrueLabel(1, train_bool = False)
            # Compose these, beware of the order.
            train_composed_quark = transforms.Compose([train_scaling, train_onehot, train_pading, train_add_quark_set])
            train_composed_gluon = transforms.Compose([train_scaling, train_onehot, train_pading, train_add_gluon_set])
            test_composed_quark = transforms.Compose([test_scaling, test_onehot, test_pading, test_add_quark_set])
            test_composed_gluon = transforms.Compose([test_scaling, test_onehot, test_pading, test_add_gluon_set])
            #composed = transforms.Compose([scaling, onehot])
            # Create a JuniprDataset class (inheriting from the PyTorch dataset class) with the data contained in self.data_path
            gluon_dataset_train = JuniprDataset(self.data_path_gluon+"_train.json", train_bool = True, transform = train_composed_gluon)
            gluon_dataset_test  = JuniprDataset(self.data_path_gluon+"_test.json", train_bool = False, transform = test_composed_gluon)
            quark_dataset_train = JuniprDataset(self.data_path_quark+"_train.json", train_bool = True, transform = train_composed_quark)
            quark_dataset_test  = JuniprDataset(self.data_path_quark+"_test.json", train_bool = False, transform = test_composed_quark)

            # Force the datasets to have as many gluons as quarks for both training and testing
            need_train_matching_bool = False
            need_test_matching_bool  = False
            if len(gluon_dataset_train) != len(quark_dataset_train):
                need_train_matching_bool = True
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

            min_test_size = int(len(gluon_dataset_test))
            # They should be equilibrated now. Cutting the test sets into validations and true tests
            val_size = int(min_test_size * self.validation_size)
            test_size = min_test_size - val_size
            quark_val_set, quark_test_set = random_split(quark_dataset_test, [val_size, test_size])
            gluon_val_set, gluon_test_set = random_split(gluon_dataset_test, [val_size, test_size])

            # Concatenate these datasets now. Remark: it is going to be imperative to shuffle them for the training dataloader (otherwise you'll first read all quarks and then all gluons). Might also need to shuffle them when assessing (otherwise you'll only plot the x first jets which will be quarks.
            self.train_set = torch.utils.data.ConcatDataset([quark_dataset_train, gluon_dataset_train])
            self.val_set   = torch.utils.data.ConcatDataset([quark_val_set, gluon_val_set])
            self.test_set  = torch.utils.data.ConcatDataset([quark_test_set, gluon_test_set])
        else:
            print("Start reading data for unary Junipr")
            # Set up the transforms to apply to the dataset
            #composed = transforms.Compose([scaling, onehot])
            
            #train_pading = PadTensors(self.padding_size, self.padding_value, train_bool = True)
            train_pading = PadTensorsWithMask(self.padding_size, self.padding_value, train_bool = True)
            train_scaling = FeatureScalingOwn(train_bool = True)
            train_onehot = GranulariseBranchings(self.granularity, train_bool = True)
            test_pading = PadTensorsWithMask(self.padding_size, self.padding_value, train_bool = False)
            test_scaling = FeatureScalingOwn(train_bool = False)
            self.used_test_scaling = test_scaling
            test_onehot = GranulariseBranchings(self.granularity, train_bool = False)

            # Compose these, beware of the order.
            train_composed = transforms.Compose([train_scaling, train_onehot, train_pading])
            test_composed  = transforms.Compose([test_scaling, test_onehot, test_pading])
            """
            test_pading = PadTensors(self.padding_size, self.padding_value, train_bool = False)
            test_scaling = FeatureScaling(self.feature_scaling_params, train_bool = False)
            test_onehot = GranulariseBranchings(self.granularity, train_bool = False)
            test_composed  = transforms.Compose([test_scaling, test_onehot, test_pading])
            # Create a JuniprDataset class (inheriting from the PyTorch dataset class) with the data contained in self.data_path
            dataset = JuniprDataset(self.data_path, train_bool = False, transform = test_composed)
            
            # Cut this dataset into two parts: train and test (note that a seed for PyTorch is set in the main so the randomness of the operation is controlled)
            dataset_size_used = int(len(dataset) * self.fraction_data)
            test_size = int(dataset_size_used * self.test_size)
            train_size = dataset_size_used - test_size
            self.train_set, self.test_set = random_split(dataset, [train_size, test_size])
            
            test_dataset_size_used = int(len(self.test_set))
            val_size = int(test_dataset_size_used * self.validation_size)
            test_size = test_dataset_size_used - val_size
            val_set, test_set = random_split(self.test_set, [val_size, test_size])
            self.train_set, self.val_set, self.test_set = self.train_set, val_set, test_set
            """
            
            # Create a JuniprDataset class (inheriting from the PyTorch dataset class) with the data contained in self.data_path
            dataset_train = JuniprDataset(self.data_path+"_train.json", train_bool = True, transform = train_composed)
            dataset_test = JuniprDataset(self.data_path+"_test.json", train_bool = False, transform = test_composed)
            
            test_dataset_size_used = int(len(dataset_test))
            val_size = int(test_dataset_size_used * self.validation_size)
            test_size = test_dataset_size_used - val_size
            val_set, test_set = random_split(dataset_test, [val_size, test_size])
            
            self.train_set, self.val_set, self.test_set = dataset_train, val_set, test_set
            
            """
            # Instantiate two dataloaders: for train and test
            self.train_dataloader = DataLoader(train_set, batch_size=2, shuffle=True)
            self.test_dataloader  = DataLoader(test_set,  batch_size=2, shuffle=True)
            """
        print("Side of train: ", len(self.train_set))
        print("Side of validation: ", len(self.val_set))
        print("Side of test: ", len(self.test_set))
        return self.train_set, self.val_set, self.test_set







