#############################################################################
#
# JuniprLargeDataset.py
#
# A class inheriting from PyTorch class to create a dataset for junipr jets
#
# Author -- Maxence Draguet (14/08/2020)
#
#############################################################################
import os
import sys
import json
from typing import Dict
import random

import numpy as np
import pandas as pd
#pd.set_option('display.max_rows', None)
from multiprocessing import Manager

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class JuniprLargeDataset(Dataset):
    """
    The Junipr Dataset class for large samples. Workes on chunks of the data. One chunk is made to contain several batch so that the chunk can be loaded into memory and kept there for as long as needed.
    If all later defined transform are applied, output of the dataset should be batchable
    
    We have: (sizes are B for batch X ...)
        - label: B int
        - multiplicity: B int
        - n_branchings: B int
        - seed_momentum: B X 4 of float
        - ending: B X default_size of int
        - mother_id_energy: B X default_size X default_size of float (values 0/1)
        - branching: B X default_size X (granularity * 4) of one hot vector int (often 40 for the last size)
        - mother_momenta: B X default_size X 4 of float
        - daughter_momenta:  B X default_size X 4 of float
        
    default_size is the (padded) size for recurrence. In the models, recurrence will be interrupted based on n_branching (which indicates the true size)
                 This default_size has to be larger than the largest branching !
    granularity is the size of the binning of branching info into x (variables are forced to take values between 0 and 1). This is followed by one-hot encoding.
    """
    
    def __init__(self, json_file_list, seed, train_bool, transform, manage_with = None, binary_junipr = False):#config: Dict):
        """
        Receives the json_file_list with the data to process as well as the chunk size (in a tuple (chunk_size, path).
        Transform is an option to be applied to samples (to scale, pad and modify them).
        
        binary_junipr is to be set True for a binary junipr large dataset.
        It only accept data with as many chunks for quarks as for gluons (note that if they must be balanced, it has to be enforced at the processing level).
        """
        super().__init__()
        random.seed(seed)
        self.manager_to_share_data = manage_with
        self.train_bool = train_bool
        self.transform = transform
        self.item_idx_to_change_chunk = dict()
        self.binary_junipr = binary_junipr
    
        self.whole_data_size = 0
        if self.binary_junipr:
            self.quark_json_file_list, self.gluon_json_file_list =json_file_list
            print("For quarks, here are the chunks: ", self.quark_json_file_list)
            print("For gluons, here are the chunks: ", self.gluon_json_file_list)
            for count in range(len(self.quark_json_file_list)):
                quark_chunk_size, quark_chunk_path = self.quark_json_file_list[count]
                gluon_chunk_size, gluon_chunk_path = self.gluon_json_file_list[count]
                self.item_idx_to_change_chunk[self.whole_data_size] = (quark_chunk_path, gluon_chunk_path)
                self.whole_data_size += quark_chunk_size + gluon_chunk_size
        else:
            self.json_file_list = json_file_list
            print(self.json_file_list)
            for count, (chunk_size, _) in enumerate(self.json_file_list):
                self.item_idx_to_change_chunk[self.whole_data_size] = self.json_file_list[count][1]
                self.whole_data_size += chunk_size
        
        print(self.item_idx_to_change_chunk)
                
        self.idx_shift_to_chunk = 0
        self.only_a_single_chunk = False
        if len(self.item_idx_to_change_chunk) == 1:
            self.only_a_single_chunk = True
        
        if self.only_a_single_chunk:
            if self.binary_junipr:
                quark_chunk_to_open, gluon_chunk_to_open  = self.item_idx_to_change_chunk[0]
                print("#############################################")
                print("\nIndex {}, Loading the single quark chunk from {}".format(0, quark_chunk_to_open))
                print("Index {}, Loading the single  gluon chunk from {}\n".format(0, gluon_chunk_to_open))
                print("#############################################")
                
                # load
                with open(quark_chunk_to_open, 'r') as json_file:
                    quark_data_array = json.load(json_file)['JuniprJets']
                with open(gluon_chunk_to_open, 'r') as json_file:
                    gluon_data_array = json.load(json_file)['JuniprJets']
                
                quark_data_array.extend(gluon_data_array)
                random.shuffle(quark_data_array)
                if self.manager_to_share_data:
                    self.chunk_in_memory = self.manager_to_share_data.list(quark_data_array)
                else:
                    self.chunk_in_memory = quark_data_array
            else:
                chunk_to_open = self.item_idx_to_change_chunk[0]
                print("#############################################")
                print("\nIndex {}, Loading the single chunk from {}\n".format(0, chunk_to_open))
                print("#############################################")
                with open(chunk_to_open, 'r') as json_file:
                    if self.manager_to_share_data:
                        data_array = json.load(json_file)['JuniprJets']
                        random.shuffle(data_array)  # inplace operation !
                        self.chunk_in_memory = self.manager_to_share_data.list(data_array)
                    else:
                        data_array = json.load(json_file)['JuniprJets']
                        random.shuffle(data_array)  # inplace operation !
            self.chunk_in_memory = data_array # a list of dictionnaries that is suffled
        
        
    def __len__(self):
        return self.whole_data_size

    def __getitem__(self, idx):
        """
        Redefine the data accessor so that doing JuniprDataset[i] returns the i-th samples of the json files with properties listed at the end.
        It requires to sometime load one (two) new chunk(s) and put it (them) in storage, which is a costly move.
        """
        first_idx = idx
        if torch.is_tensor(idx):
            idx = idx.tolist()
            first_idx = idx[0]
    
        if not(self.only_a_single_chunk) and (first_idx in self.item_idx_to_change_chunk):
            # This means you've reached the start of a new chunk that has to be loaded into memory
            if self.binary_junipr:
                quark_chunk_to_open, gluon_chunk_to_open  = self.item_idx_to_change_chunk[first_idx]
                print("#############################################")
                print("\nIndex {}, Loading a new quark chunk from {}".format(first_idx, quark_chunk_to_open))
                print("Index {}, Loading a new gluon chunk from {}\n".format(first_idx, gluon_chunk_to_open))
                print("#############################################")
                
                # load
                with open(quark_chunk_to_open, 'r') as json_file:
                    quark_data_array = json.load(json_file)['JuniprJets']
                with open(gluon_chunk_to_open, 'r') as json_file:
                    gluon_data_array = json.load(json_file)['JuniprJets']

                quark_data_array.extend(gluon_data_array)
                random.shuffle(quark_data_array)
                if self.manager_to_share_data:
                    self.chunk_in_memory = self.manager_to_share_data.list(quark_data_array)
                else:
                    self.chunk_in_memory = quark_data_array
            else:
                chunk_to_open = self.item_idx_to_change_chunk[first_idx]
                print("#############################################")
                print("\nIndex {}, Loading a new chunk from {}\n".format(first_idx, chunk_to_open))
                print("#############################################")
                with open(chunk_to_open, 'r') as json_file:
                    if self.manager_to_share_data:
                        data_array = json.load(json_file)['JuniprJets']
                        random.shuffle(data_array)  # inplace operation !
                        self.chunk_in_memory = self.manager_to_share_data.list(data_array)
                    else:
                        data_array = json.load(json_file)['JuniprJets']
                        random.shuffle(data_array)  # inplace operation !
                        self.chunk_in_memory = data_array # a list of dictionnaries that is suffled
            self.idx_shift_to_chunk = first_idx
                
        if type(idx) == list:
            idx = [old_idx - self.idx_shift_to_chunk for old_idx in idx]
        else:
            idx = idx - self.idx_shift_to_chunk

        targeted_jet = self.chunk_in_memory[idx]  # returns the associated dictionnary

        label            = targeted_jet["label"]
        n_branchings     = targeted_jet["n_branchings"]
        seed_momentum    = torch.FloatTensor(targeted_jet["seed_momentum"])
        if not(self.train_bool):
            CSJets           = torch.FloatTensor(targeted_jet["CSJets"])
            CS_ID_mothers    = torch.IntTensor(targeted_jet["CS_ID_mothers"])
            CS_ID_daugthers  = torch.IntTensor([[d[0], d[1]] for d in targeted_jet["CS_ID_daugthers"]])
        mother_id_energy = torch.IntTensor(targeted_jet["mother_id_energy_order"])
        branching        = torch.FloatTensor(targeted_jet["branching"])
        mother_momenta   = torch.FloatTensor(targeted_jet["mother_momenta"])
        daughter_momenta = torch.FloatTensor([np.concatenate([d[0], d[1]]) for d in targeted_jet["daughter_momenta"]])

        if self.train_bool:
            sample = {
                      "label": label,
                      "n_branchings": n_branchings,
                      "seed_momentum": seed_momentum,
                      "mother_id_energy": mother_id_energy,
                      "branching": branching,
                      "mother_momenta": mother_momenta,
                      "daughter_momenta": daughter_momenta
                    }
        else:
            sample = {
                    "label": label,
                    "n_branchings": n_branchings,
                    "seed_momentum": seed_momentum,
                    "mother_id_energy": mother_id_energy,
                    "CSJets": CSJets,
                    "CS_ID_mothers": CS_ID_mothers,
                    "CS_ID_daugthers": CS_ID_daugthers,
                    "branching": branching,
                    "unscaled_branching": branching,
                    "mother_momenta": mother_momenta,
                    "daughter_momenta": daughter_momenta
                    }

        if self.transform:
            sample = self.transform(sample)
        return sample
