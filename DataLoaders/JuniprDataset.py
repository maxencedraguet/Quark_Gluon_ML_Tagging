#############################################################################
#
# JuniprDataset.py
#
# A class inheriting from PyTorch class to create a dataset for junipr jets
#
# Author -- Maxence Draguet (06/07/2020)
#
#############################################################################
import os
import sys
import json
from typing import Dict

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

EPSILON = 1e-10
INF     = 1e8

class JuniprDataset(Dataset):
    """
    The Junipr Dataset class
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
    
    def __init__(self, json_file, transform):#config: Dict):
        """
        Receives the json_file with the data to process as well as the root directory
        Transform is an option to be applied to samples (to scale, pad and modify them).
        """
        super().__init__()
        #self.json_file = config.get(["JuniprDataset", "data_path"])
        self.json_file = json_file
        with open(self.json_file) as json_file:
            self.data_array = json.load(json_file)['JuniprJets'] #a list of dictionnaries
        self.data_size = len(self.data_array)
        #self.transform = config.get(["JuniprDataset", "transform"])
        self.transform = transform
        #self.padding_to_size =config.get(["JuniprDataset", "pad_to_size"]) #
        

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        """
        Redefine the data accessor so that doing JuniprDataset[i]
        returns the i-th samples of the json files with properties listed at the end.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        targeted_jet = self.data_array[idx] #returns the associated dictionnary
        #.... do something to get samples from the jet dictionary
        label            = targeted_jet["label"]
        multiplicity     = targeted_jet["multiplicity"]
        n_branchings     = targeted_jet["n_branchings"]
        seed_momentum    = torch.FloatTensor(targeted_jet["seed_momentum"])
        # return a tensor of size number_of_jets*size of a list
        #CSJets           = torch.FloatTensor(targeted_jet["CSJets"])
        # return a tensor of size number_of_branching*size of a branching
        ending = torch.IntTensor()
        mother_id_energy = targeted_jet["mother_id_energy_order"]
        branching        = torch.FloatTensor(targeted_jet["branching"])
        mother_momenta   = torch.FloatTensor(targeted_jet["mother_momenta"])
        # return a tensor of size number_of_branching* 2 * size of a branching
        daughter_momenta = torch.FloatTensor([np.concatenate([d[0], d[1]]) for d in targeted_jet["daughter_momenta"]])
        
        sample = {
                "label": label,
                "multiplicity": multiplicity,
                "n_branchings": n_branchings,
                "seed_momentum": seed_momentum,
                "ending": ending,
                "mother_id_energy": mother_id_energy,
                #"CSJets": CSJets,
                "branching": branching,
                "mother_momenta": mother_momenta,
                "daughter_momenta": daughter_momenta
                }

        if self.transform:
            sample = self.transform(sample)
        return sample


class PadTensors(object):
    """
    Pads the tensors of the sample to have the default_size
    The tensors CSJets, branching, mother_momenta, and daughter_momenta will be padded to have size pad_to_size from
    
    Note that n_branching will keep the real size of these tensors
    """
    def __init__(self, default_size, pad_token):
        assert isinstance(default_size, int)
        self.default_size = default_size
        self.pad_token = pad_token
            
    def __call__(self, sample):
        branching, mother_id_energy = sample["branching"], sample["mother_id_energy"]
        mother_momenta, daughter_momenta = sample["mother_momenta"], sample["daughter_momenta"]
        
        padded_branching = np.ones((self.default_size, branching.size()[-1])) * self.pad_token
        padded_mother_momenta = np.ones((self.default_size, mother_momenta.size()[-1])) * self.pad_token
        padded_daughter_momenta = np.ones((self.default_size, daughter_momenta.size()[-1])) * self.pad_token

        padded_branching[:branching.size()[0] , :] = branching
        padded_mother_momenta[:mother_momenta.size()[0] , :] = mother_momenta
        padded_daughter_momenta[:daughter_momenta.size()[0] , :] = daughter_momenta
        
        # Case of mother_id_energy: particular
        #   - has to be padded to self.output_size * self.output_size (first for recurrence, second for possible mothers)
        #   - the second dimension is actually a one-hot encoding of the index of the mother (in energy order, 0 being most energetic)
        # Hijack the process to also produced the ending tensor (size: recurrence * 1 with value 0 if going on and 1 when stop
        padded_mother_id = np.ones((self.default_size, self.default_size)) * self.pad_token
        ending_val = np.ones((self.default_size)) * self.pad_token
        for counter, elem in enumerate(mother_id_energy):
            padded_mother_id[counter, elem] = 1.0
            ending_val[counter] = 0.0
        ending_val[len(mother_id_energy)] = 1.0
        
        return { "label": sample["label"],
                 "multiplicity": sample["multiplicity"],
                 "n_branchings": sample["n_branchings"],
                 "seed_momentum": sample["seed_momentum"],
                 "ending": ending_val,
                 "mother_id_energy": padded_mother_id,
                 #"CSJets": sample["CSJets"],
                 "branching": torch.FloatTensor(padded_branching),
                 "mother_momenta": torch.FloatTensor(padded_mother_momenta),
                 "daughter_momenta": torch.FloatTensor(padded_daughter_momenta)
                }

class FeatureScaling(object):
    """
    Scales the jet values as presented in Junipr paper. Note that the pertinence of this here is reduced since
    our dataset is not centred around some jet values (expect for R_jet and R_sub). The procedure is however kept as
    not other scaling feature seemed natural
    
    Major issue: the branching values really need to between 0 and 1. They are indeed to become one-hot vectors and
    the discretisation only work if these values are in the right range.
    E_JET = 500.0
    E_SUB = 0.1 (much lower not to get negative component.
    R_JET = np.pi / 2
    R_SUB = 0.1
    """
    def __init__(self, feature_parameter):
        E_jet = feature_parameter[0]
        E_sub = feature_parameter[1]
        R_jet = feature_parameter[2]
        R_sub = feature_parameter[3]
        
        # For momenta (any sort of 4 momenta)
        self.mom_e_shift = np.log(E_sub)
        self.mom_e_scale = np.log(E_jet) - self.mom_e_shift
        self.mom_th_shift = np.log(E_sub * R_sub / E_jet)
        self.mom_th_scale = np.log(R_jet) - self.mom_th_shift
        self.mom_phi_shift = 0
        self.mom_phi_scale = 2 * np.pi - self.mom_phi_shift
        self.mom_mass_shift = np.log(E_sub)
        self.mom_mass_scale = np.log(E_jet) - self.mom_e_shift
        
        # For branches (note: as for momenta for phi variable)
        self.branch_z_shift = np.log(E_sub / E_jet)
        self.branch_z_scale = np.log(0.5) - self.branch_z_shift
        self.branch_theta_shift = np.log(R_sub / 2.0)
        self.branch_theta_scale = np.log(R_jet) - self.branch_theta_shift
        self.branch_phi_shift = 0
        self.branch_phi_scale = 2 * np.pi - self.mom_phi_shift
        self.branch_delta_shift = np.log(E_sub * R_sub / E_jet)
        self.branch_delta_scale = np.log(R_jet / 2.0) - self.branch_delta_shift
    
    def __call__(self, sample):
        branching, mother_momenta, daughter_momenta = sample["branching"], sample["mother_momenta"], sample["daughter_momenta"]
        seed_momentum = sample["seed_momentum"]
        # operate on branching: z, theta, phi, delta
        branching[:, 0] = (np.log(np.clip(branching[:, 0], EPSILON, INF)) - self.branch_z_shift) / self.branch_z_scale
        branching[:, 1] = (np.log(np.clip(branching[:, 1], EPSILON, INF)) - self.branch_theta_shift) / self.branch_theta_scale
        branching[:, 2] = (branching[:, 2] -  self.branch_phi_shift) / self.branch_phi_scale
        branching[:, 3] = (np.log(np.clip(branching[:, 3], EPSILON, INF)) - self.branch_delta_shift) / self.branch_delta_scale
        #these two tests should be REMOVED in final version
        if(branching[branching<0].size()[0] !=0):
            print("Negative values in branching")
        if(branching[branching>1].size()[0] !=0):
            print("Values above 1 in branching")
        
        # operate on each momenta (mother and daughter)
        # Mother:
        mother_momenta[:, 0] = (np.log(np.clip(mother_momenta[:, 0], EPSILON, INF)) - self.mom_e_shift) / self.mom_e_scale
        mother_momenta[:, 1] = (np.log(np.clip(mother_momenta[:, 1], EPSILON, INF)) - self.mom_th_shift) / self.mom_th_scale
        mother_momenta[:, 2] = (mother_momenta[:, 2] - self.mom_phi_shift) / self.mom_phi_scale
        mother_momenta[:, 3] = (np.log(np.clip(mother_momenta[:, 3], EPSILON, INF)) - self.mom_mass_shift) / self.mom_mass_scale
        
        #daugther1
        daughter_momenta[:, 0] = (np.log(np.clip(daughter_momenta[:, 0], EPSILON, INF)) - self.mom_e_shift) / self.mom_e_scale
        daughter_momenta[:, 1] = (np.log(np.clip(daughter_momenta[:, 1], EPSILON, INF)) - self.mom_th_shift) / self.mom_th_scale
        daughter_momenta[:, 2] = (daughter_momenta[:, 2] - self.mom_phi_shift) / self.mom_phi_scale
        daughter_momenta[:, 3] = (np.log(np.clip(daughter_momenta[:, 3], EPSILON, INF)) - self.mom_mass_shift) / self.mom_mass_scale
        
        #daughter2
        daughter_momenta[:, 4] = (np.log(np.clip(daughter_momenta[:, 4], EPSILON, INF)) - self.mom_e_shift) / self.mom_e_scale
        daughter_momenta[:, 5] = (np.log(np.clip(daughter_momenta[:, 5], EPSILON, INF)) - self.mom_th_shift) / self.mom_th_scale
        daughter_momenta[:, 6] = (daughter_momenta[:, 6] - self.mom_phi_shift) / self.mom_phi_scale
        daughter_momenta[:, 7] = (np.log(np.clip(daughter_momenta[:, 7], EPSILON, INF)) - self.mom_mass_shift) / self.mom_mass_scale

        #seed_momentum
        seed_momentum[0] = (np.log(np.clip( seed_momentum[0], EPSILON, INF)) - self.mom_e_shift) / self.mom_e_scale
        seed_momentum[1] = (np.log(np.clip(seed_momentum[1], EPSILON, INF)) - self.mom_th_shift) / self.mom_th_scale
        seed_momentum[2] = (seed_momentum[2] - self.mom_phi_shift) / self.mom_phi_scale
        seed_momentum[3] = (np.log(np.clip(seed_momentum[3], EPSILON, INF)) - self.mom_mass_shift) / self.mom_mass_scale

        return { "label": sample["label"],
                 "multiplicity": sample["multiplicity"],
                 "n_branchings": sample["n_branchings"],
                 "seed_momentum": seed_momentum,
                 "ending": sample["ending"],
                 "mother_id_energy": sample["mother_id_energy"],
                 #"CSJets": sample["CSJets"],
                 "branching": branching,
                 "mother_momenta": mother_momenta,
                 "daughter_momenta": daughter_momenta
                }

class OneHotBranch(object):
    """
    Converts the branching infos into one hot vector of given granularity
    """
    def __init__(self, granularity):
        self.granularity = granularity
    
    def __call__(self, sample):
        branching = sample["branching"]
        # quick fix to set negative values to 0 and above 1 to (just below) 1. Normally, the parameters have been chosen so that should mostly never be necessary

        branching = np.clip(branching*self.granularity, 0, self.granularity-1).int()

        y = torch.eye(10) #this has 10 entries: the one hot encoding of int value 0 to 9
        #torch.cat([y[0], y[1], y[2], y[3]], dim = 0)
        branching_dis = torch.empty(branching.size()[0], branching.size()[1]* self.granularity, dtype=torch.int)
        for row in range(branching.size()[0]):
            branching_dis[row, :] = torch.cat([y[branching[row, 0].item()],
                                               y[branching[row, 1].item()],
                                               y[branching[row, 2].item()],
                                               y[branching[row, 3].item()]], dim = 0)
        return {"label": sample["label"],
                "multiplicity": sample["multiplicity"],
                "n_branchings": sample["n_branchings"],
                "seed_momentum": sample["seed_momentum"],
                "mother_id_energy": sample["mother_id_energy"],
                "ending": sample["ending"],
                #"CSJets": sample["CSJets"],
                "branching": branching_dis,
                "mother_momenta": sample["mother_momenta"],
                "daughter_momenta": sample["daughter_momenta"]
               }

