#############################################################################
#
# RecurrentNetwork.py
#
# A recurrent network using PyTorch
#
# Author -- Maxence Draguet (08/07/2020)
#
#############################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseNetwork import _BaseNetwork
from .Networks import NeuralNetwork

class RecurrentNetwork(_BaseNetwork):
    
    def __init__(self, config: Dict):
        self.extract_parameters(config)
        _BaseNetwork.__init__(self, config=config)
        
        # Method inherited from mother class:
        self.nonlinearity = self.identify_nonlinfunc(self.nonlinearity_name)
        
        self.construct_layers()
    
    def extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        dimensions should be a list of successive size.
        """
        self.input_size = config.get(["Junipr_Model", "Structure", "Recurrent", "input_dimensions"])
        self.hidden_size = config.get(["Junipr_Model", "Structure", "Recurrent", "hidden_dimensions"])
        self.output_size = config.get(["Junipr_Model", "Structure", "Recurrent", "output_dimensions"])
        self.initialisation = config.get(["Junipr_Model", "Structure", "Recurrent", "initialisation"]) #be sure it's Xavier Normal
        self.nonlinearity_name = config.get(["Junipr_Model", "Structure", "Recurrent", "nonlinearity"])
    
    def initialise_seed_momenta_to_hidden(self, config: Dict):
        """
        A first pass over a NNetwork to transform seed_momenta into initial hidden state.
        """
        self.seed_momenta_to_hidden_network = NeuralNetwork(source = "RecurrentInit", config=config)
        
    def construct_layers(self, config: Dict) -> None:
        """
        Constructs the network according to specified structure
        """
        # This first network is for the seed_momenta. A pass over this should be done only before the first recurrence starts
        self.initialise_seed_momenta_to_hidden(config)
        
        # The recurrent one:
        self.recurrent_network = self.initialise_weights(nn.Linear(self.input_size + self.hidden_size, self.hidden_size))

    def forward(self, input, hidden) -> torch.Tensor:
        """
        Pass the inputs and the hidden through self.recurrent_network and applies the nonlinearity (typically tanh)s.
        """
        combined = torch.cat((input, hidden), 1)
        hidden = self.nonlinearity(self.recurrent_network(combined))
        
        return hidden

    def save_model(self, path) -> None:
        torch.save(self.state_dict(), os.path.join(path, 'saved_RNN_weights.pt'))
    
    def load_model(self, path) -> None:
        """
        Load saved weights into model.
        """
        saved_weights = torch.load(path)
        self.load_state_dict(saved_weights)
