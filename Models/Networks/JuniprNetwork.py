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
        self.padding_size = config.get(["Junipr_Model", "Junipr_Dataset", "padding_size"])
    
    
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
        # To remove mothers not to be considered:
        lower_triangular_ones = torch.tril(torch.tensor(np.ones((self.padding_size, self.padding_size)))).float()
        tensor_dismiss_impossible_mothers = torch.where(lower_triangular_ones != 0, torch.tensor([0.0]), torch.tensor([float('-inf')]))
        self.dismiss_impossible_mothers = tensor_dismiss_impossible_mothers[None, :, :].detach()
        
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
        last_daughters_momenta = input["last_daughters_momenta"]
        mother_momenta      = input["mother_momenta"]
        n_branching         = input["n_branchings"]
        
        #print("daughters_momenta \n", daughters_momenta.size())
        #print("last_daughters_momenta \n", last_daughters_momenta.size())
        #print("mother_momenta \n", mother_momenta.size())
        
        #print("In junipr network, size of daughters_momenta: {} and full entries \n {}".format(daughters_momenta.size(), daughters_momenta))
        #print("In junipr network, size of daughters_momenta: {}".format(daughters_momenta.size()))
        # Use n_branching-1 since the last daughters have been set aside in last_daughters_momenta
        daughters_momenta_PACK = rnn.pack_padded_sequence(daughters_momenta, n_branching-1, batch_first=True, enforce_sorted = False)
        # As a weird effect of interleaving the tensor entries (batch_sample1, batch_sample2, ... batch_sampleX, and again)
        
        first_hidden_states_unshaped = self.seed_momenta_to_hidden_network(seed_momenta)
        #print("first_hidden_states_unshaped: ", first_hidden_states_unshaped.grad_fn)
        # Needs a dimension tweak
        first_hidden_states = first_hidden_states_unshaped[None, :, :]
        #print("first_hidden_states_unshaped: ", first_hidden_states_unshaped.grad_fn)
        #print("In junipr network, size of first_hidden_states: {}".format(first_hidden_states.size()))
        if self.RNN_type == "lstm":
            first_c_states = self.seed_momenta_to_c0_network(seed_momenta)
            first_c_states = first_c_states[None, :, :]
            hidden_states_packed, (last_hiddens, last_c_states) = self.recurrent_network(daughters_momenta_PACK, first_hidden_states, first_c_states)
        else:
            hidden_states_packed, last_hiddens = self.recurrent_network(daughters_momenta_PACK, first_hidden_states)
        #print("\nIn junipr network, hidden output: {}\n".format(hidden_states))
        #print("In junipr network, last hidden output: {}".format(last_hiddens.size()))
        #print("hidden_states_packed: ", hidden_states_packed.grad_fn)
        hidden_states_unpacked, _ = rnn.pad_packed_sequence(hidden_states_packed, batch_first=True)
        #print("hidden_states_unpacked: ", hidden_states_unpacked.grad_fn)
        #print("In junipr network, hidden output UNPACKED size: {}".format(hidden_states.size()))
        #print("first_hidden_states \n",first_hidden_states)
        #first_hidden_states_reshaped = torch.reshape(first_hidden_states, (first_hidden_states.size()[1], 1, first_hidden_states.size()[2]))
        first_hidden_states_reshaped = first_hidden_states.view(first_hidden_states.size()[1], 1, first_hidden_states.size()[2])
        #print("first_hidden_states_reshaped: ", first_hidden_states_reshaped.grad_fn)
        #print("hidden_states \n",hidden_states)
        #print("In junipr network, size of hidden_states: {}".format(hidden_states.size()))
        hidden_states = torch.cat([first_hidden_states_reshaped, hidden_states_unpacked], dim = 1)
        #print("hidden_states: ", hidden_states.grad_fn)
        
        trimmed_padding_size = hidden_states.size()[1]
        #print("hidden_states after cat \n",hidden_states)
        #print("In junipr network, size of hidden_states after cat: {}".format(hidden_states.size()))
        
        # Do a final step through the recurrence. This is to get the last hidden state for the very end probability computation
        # Take back the last configuration in order to do this.
        #print("last_daughters_momenta \n", last_daughters_momenta.size())
        last_daughters_momenta = last_daughters_momenta[:, None, :]
        #print("last_daughters_momenta: ", last_daughters_momenta.grad_fn)
        #print("last_daughters_momenta tweaked \n", last_daughters_momenta.size())
        if self.RNN_type == "lstm":
            _, (last_hidden_states_unshaped, _) = self.recurrent_network(last_daughters_momenta, last_hiddens, last_c_states)
        else:
            _, last_hidden_states_unshaped = self.recurrent_network(last_daughters_momenta, last_hiddens)
            #last_hidden_states = torch.reshape(last_hidden_states_unshaped, (last_hidden_states_unshaped.size()[1], 1, last_hidden_states_unshaped.size()[2]))
            #print("last_hidden_states_unshaped: ", last_hidden_states_unshaped.grad_fn)
        last_hidden_states = last_hidden_states_unshaped.view(last_hidden_states_unshaped.size()[1], 1, last_hidden_states_unshaped.size()[2])
        #print("last_hidden_states: ", last_hidden_states.grad_fn)
        #print("first_hidden_states : ", first_hidden_states.size())
        #print("hidden_states : ", hidden_states.size())
        #print("last_hiddens : ", last_hidden_states.size())

        #print("first_hidden_states : \n", first_hidden_states)
        #print("hidden_states : \n", hidden_states)
        # print("last_hiddens : \n", last_hidden_states)
        
        output_end = self.p_end_network(hidden_states)
        #print("In junipr network, output end branch size: {}".format(output_end.size()))
        # This is missing a treatment of the very last node (where no branching ocurs)
        output_very_end = self.p_end_network(last_hidden_states)

        #print("In junipr network, output very end branch size: {}".format(output_very_end.size()))
        #print("In junipr network, output very end branch: {}".format(output_very_end))
        # should find a way of concatenating output_end and output_very_end along the seq dimension
        
        output_mother_unbalance = self.p_mother_network(hidden_states)[:, :trimmed_padding_size, :trimmed_padding_size]
        #print("output_mother_unbalance: ", output_mother_unbalance.grad_fn)
        #lower_triangular_ones = torch.tril(torch.tensor(np.ones((trimmed_padding_size, trimmed_padding_size)))).float()
        #tensor_dismiss_impossible_mothers = torch.where(lower_triangular_ones != 0, torch.tensor([0.0]), torch.tensor([float('-inf')]))
        #tensor_dismiss_impossible_mothers = tensor_dismiss_impossible_mothers[None, :, :].detach()
        
        remove_nasty_values = self.dismiss_impossible_mothers[:, :trimmed_padding_size, :trimmed_padding_size].repeat(output_mother_unbalance.size()[0], 1, 1).detach()
        #print("remove_nasty_values: ", remove_nasty_values.grad_fn)
        #remove_nasty_values = tensor_dismiss_impossible_mothers.repeat(output_mother_unbalance.size()[0], 1, 1).detach()
        #print("output_mother_unbalance \n",output_mother_unbalance)
        output_mother = output_mother_unbalance + remove_nasty_values
        #print("output_mother: ", output_mother.grad_fn)
        #print("output_mother \n",output_mother)
    
        #print("In junipr network, output mother branch size: {}".format(output_mother.size()))
        # A small amount of processing is required for mother: there should only be as many considered entries as particles in the state.
        # This can be easily done to the batch: just apply keep the lower triangular part (with diagonal) in a matrix such that batch, recurrence, feature.
        #output_mother = torch.tril(output_mother, diagonal = 0)
        # To limit the softmax in the loss to the non-zero value (all zero-values are due to trim, absolutely unlikely for a value out of the trim to be null)
        # output_mother = torch.where(output_mother != 0, output_mother, torch.tensor([float('-inf')]))

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
            
            #print("In junipr network, combined_branch          size: {}".format(combined_branch         .size()))
            #print("In junipr network, combined_branch_with_z   size: {}".format(combined_branch_with_z  .size()))
            #print("In junipr network, combined_branch_with_ztd size: {}".format(combined_branch_with_ztd.size()))
            #print("In junipr network, combined_branch_with_zt  size: {}".format(combined_branch_with_zt .size()))
            
            
            output_branch_z = self.p_branch_z_network(combined_branch)
            output_branch_t = self.p_branch_t_network(combined_branch_with_z)
            output_branch_p = self.p_branch_p_network(combined_branch_with_ztd)
            output_branch_d = self.p_branch_d_network(combined_branch_with_zt)
            #print("In junipr network, output_branch_z size: {}".format(output_branch_z.size()))
            #print("In junipr network, output_branch_t size: {}".format(output_branch_t.size()))
            #print("In junipr network, output_branch_p size: {}".format(output_branch_p.size()))
            #print("In junipr network, output_branch_d size: {}".format(output_branch_d.size()))
        #print("output_end :      ",output_end.grad_fn)
        #print("output_very_end : ",output_very_end.grad_fn)
        #print("output_mother :   ",output_mother.grad_fn)
        #print("output_branch_z : ",output_branch_z.grad_fn)
        #print("output_branch_t : ",output_branch_t.grad_fn)
        #print("output_branch_p : ",output_branch_p.grad_fn)
        #print("output_branch_d : ",output_branch_d.grad_fn)
        
        # Join the outputs and return
        output_dictionnary = dict()
        #output_dictionnary["hidden_state"]    = hidden_states      # seems useless to return this
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

