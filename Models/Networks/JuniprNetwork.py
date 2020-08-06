#############################################################################
#
# JuniprNetwork.py
#
# A JUNIPR network implementation using PyTorch to implement the JUNIPR model
#
# Author -- Maxence Draguet (10/07/2020)
#
#############################################################################

from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from joblib import dump, load

import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F
from sklearn import metrics

from tensorboardX import SummaryWriter

from .BaseNetwork import _BaseNetwork
from .NNetwork import NeuralNetwork
from .RecurrentNetwork import RecurrentNetwork

class JuniprNetwork(_BaseNetwork):
    def __init__(self, config: Dict):
        print("In init Junipr")
        self.extract_parameters(config)
        _BaseNetwork.__init__(self, config=config)
        self.setup_Model(config)
    
    def extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class.
        Note that most parameters are in fact needed at the next level (in creating the actual networks).
        """
        # Whether to represent branch branch in 1 or 4 networks
        self.branch_treatment = config.get(["Junipr_Model", "Structure", "branch_structure"])
        self.RNN_type = config.get(["Junipr_Model", "Structure", "Recurrent", "RNN_type"])
        
        # Need to know granularity to manipulate the branch output.
        self.granularity = config.get(["Junipr_Model", "Junipr_Dataset", "granularity"])
    
    
    def setup_Model(self, config: Dict):
        """
        To set up the different components of the JUNIPR model
        """
        # Before setting the RNN, we need to transform seed_momenta into a hidden_state
        # This latter mapping will be controlled by:
        #thing = NeuralNetwork(source = "RecurrentInit", config=config)
        #self.seed_momenta_to_hidden_network =
        self.seed_momenta_to_hidden_network = NeuralNetwork(source = "RecurrentInit", config=config)
        if self.RNN_type == "lstm":
            self.seed_momenta_to_c0_network = NeuralNetwork(source = "RecurrentInit", config=config)
        
        # We can then initiate the Recurrent Network (it's first hidden state will be the ouput of self.seed_momenta_to_hidden_network
        self.recurrent_network = RecurrentNetwork(config=config)
        
        # We can then set up the various MLP taking the hidden state of the RNN to produce:
        #   - P_end
        self.p_end_network = NeuralNetwork(source = "JuniprEnd", config=config)
        
        #   - P_mother
        self.p_mother_network = NeuralNetwork(source = "JuniprMother", config=config)
        
        #   - P_branch
        if self.branch_treatment == "unique":
            self.p_branch_network = NeuralNetwork(source = "JuniprBranch", config=config)
        elif self.branch_treatment == "multiple":
            self.p_branch_z_network = NeuralNetwork(source = "JuniprBranchZ", config=config)
            self.p_branch_t_network = NeuralNetwork(source = "JuniprBranchT", config=config)
            self.p_branch_p_network = NeuralNetwork(source = "JuniprBranchP", config=config)
            self.p_branch_d_network = NeuralNetwork(source = "JuniprBranchD", config=config)
        else:
            raise ValueError("Invalid branch treatment {}".format(self.branch_treatment))

    def forward(self, input):
        """
        The heart of the JUNIPR model: what happens when it gets a sample.
        A "sample" is a weird object: it's a dictionnary of tensor. Each of these tensors
        is batched. Several information are accessible (more details in Junipr_Dataset).
        
        Might make more sense to store hidden info here.
        """
        seed_momenta        = input["seed_momentum"]
        daughters_momenta   = input["daughter_momenta"]
        mother_momenta      = input["mother_momenta"]
        n_branching         = input["n_branchings"]
        
        #print("In junipr network, size of daughters_momenta: {} and full entries \n {}".format(daughters_momenta.size(), daughters_momenta))
        #print("In junipr network, size of daughters_momenta: {}".format(daughters_momenta.size()))
        daughters_momenta_PACK = rnn.pack_padded_sequence(daughters_momenta, n_branching, batch_first=True, enforce_sorted = False)
        # As a weird effect of interleaving the tensor entries (batch_sample1, batch_sample2, ... batch_sampleX, and again)
        
        first_hidden_states = self.seed_momenta_to_hidden_network(seed_momenta)
        # Needs a dimension tweak
        first_hidden_states = first_hidden_states[None, :, :]
        #print("In junipr network, size of first_hidden_states: {}".format(first_hidden_states.size()))
        if self.RNN_type == "lstm":
            first_c_states = self.seed_momenta_to_c0_network(seed_momenta)
            first_c_states = first_c_states[None, :, :]
        
        if self.RNN_type == "lstm":
            hidden_states, last_hiddens = self.recurrent_network(daughters_momenta_PACK, first_hidden_states, first_c_states)
        else:
            hidden_states, last_hiddens = self.recurrent_network(daughters_momenta_PACK, first_hidden_states)
        #print("\nIn junipr network, hidden output: {}\n".format(hidden_states))
        #print("In junipr network, last hidden output: {}".format(last_hiddens.size()))
        hidden_states, _ = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        #print("In junipr network, hidden output UNPACKED size: {}".format(hidden_states.size()))
        
        output_end = self.p_end_network(hidden_states)
        #print("In junipr network, output end branch size: {}".format(output_end.size()))
        # This is missing a treatment of the very last node (where no branching ocurs)
        output_very_end = self.p_end_network(last_hiddens)
        #print("In junipr network, output very end branch size: {}".format(output_very_end.size()))
        #print("In junipr network, output very end branch: {}".format(output_very_end))
        # should find a way of concatenating output_end and output_very_end along the seq dimension
        
        output_mother = self.p_mother_network(hidden_states)
        #print("In junipr network, output mother branch size: {}".format(output_mother.size()))
        # A small amount of processing is required for mother: there should only be as many considered entries as particles in the state.
        # This can be easily done to the batch: just apply keep the lower triangular part (with diagonal) in a matrix such that batch, recurrence, feature.
        output_mother = torch.tril(output_mother, diagonal = 0)
        # To limit the softmax in the loss to the non-zero value (all zero-values are due to trim, absolutely unlikely for a value out of the trim to be null)
        output_mother = torch.where(output_mother != 0, output_mother, torch.tensor([float('-inf')]))

        trimmed_padding_size = hidden_states.size()[1]
        # Branch
        #print("In junipr network, mother_momenta size: {}".format(mother_momenta.size()))
        mother_momenta = mother_momenta[:, :trimmed_padding_size, :]
        #print("In junipr network, mother_momenta size changed: {}".format(mother_momenta.size()))
        combined_branch = torch.cat((hidden_states, mother_momenta), dim = 2)
        #print("In junipr network, combined_branch size: {}".format(combined_branch.size()))
        if self.branch_treatment == "unique":
            output_branch = self.p_branch_network(combined_branch)
            #print("In junipr network, output_branch size: {}".format(output_branch.size()))
            output_branch_z = output_branch[:, :, :self.granularity]
            output_branch_t = output_branch[:, :, self.granularity:self.granularity*2]
            output_branch_p = output_branch[:, :, self.granularity*2:self.granularity*3]
            output_branch_d = output_branch[:, :, self.granularity*3:]
        
        elif self.branch_treatment == "multiple":
            #print("In multiple")
            # In this case, you have different options on what to feed to each network.
            # The implemented version is to have sequentially increasing input for z, theta, delta and phi
            # This could be modified
            branch_input_z = input["branching"][:, :trimmed_padding_size, 0]
            branch_input_t = input["branching"][:, :trimmed_padding_size, 1]
            branch_input_d = input["branching"][:, :trimmed_padding_size, 3]
            
            # Reshape these
            branch_input_z = branch_input_z[:, :, None]
            branch_input_t = branch_input_t[:, :, None]
            branch_input_d = branch_input_d[:, :, None]
            #print("In junipr network, branch_input_d size: {}".format(branch_input_d.size()))
            #print("In junipr network, combined_branch size: {}".format(combined_branch.size()))
            combined_branch_with_z   =  torch.cat((combined_branch, branch_input_z), dim = 2)
            #print("Size ", combined_branch_with_z.size())
            combined_branch_with_zt  =  torch.cat((combined_branch_with_z, branch_input_t), dim = 2)
            #print("Size ", combined_branch_with_zt.size())
            combined_branch_with_ztd =  torch.cat((combined_branch_with_zt, branch_input_d), dim = 2)
            
            output_branch_z = self.p_branch_z_network(combined_branch)
            output_branch_t = self.p_branch_t_network(combined_branch_with_z)
            output_branch_p = self.p_branch_p_network(combined_branch_with_ztd)
            output_branch_d = self.p_branch_d_network(combined_branch_with_zt)
        
        # Join the outputs and return
        output_dictionnary = dict()
        output_dictionnary["hidden_state"]    = hidden_states
        output_dictionnary["output_end"]      = output_end
        output_dictionnary["output_very_end"] = output_very_end
        output_dictionnary["output_mother"]   = output_mother
        output_dictionnary["output_branch_z"] = output_branch_z
        output_dictionnary["output_branch_t"] = output_branch_t
        output_dictionnary["output_branch_p"] = output_branch_p
        output_dictionnary["output_branch_d"] = output_branch_d
        
        return output_dictionnary
            
    def save_model(self, path) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, 'saved_JUNIPR_weights.pt'))
    
    def load_model(self, path) -> None:
        """
        Load saved weights into model.
        """
        saved_weights = torch.load(path)
        self.load_state_dict(saved_weights)

