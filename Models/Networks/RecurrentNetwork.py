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
from .NNetwork import NeuralNetwork

class RecurrentNetwork(_BaseNetwork):
    
    def __init__(self, config: Dict):
        self.extract_parameters(config)
        
        # Method inherited from mother class:
        self.nonlinearity = self.identify_nonlinfunc(self.nonlinearity_name)
        _BaseNetwork.__init__(self, config=config)
        self.construct_layers()
    
    def extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        dimensions should be a list of successive size.
        """
        self.RNN_type = config.get(["Junipr_Model", "Structure", "Recurrent", "RNN_type"])
        self.input_size = config.get(["Junipr_Model", "Structure", "Recurrent", "input_dimensions"])
        self.hidden_size = config.get(["Junipr_Model", "Structure", "Recurrent", "hidden_dimensions"])
        self.nonlinearity_name = config.get(["Junipr_Model", "Structure", "Recurrent", "nonlinearity"])
        self.end_nonlinearity_name = "identity"
    
    def construct_layers(self) -> None:
        """
        Constructs the top element. Note:initialisaiton of weights is forced to be Xavier uniform. This is a hotly debated topic but this should not be overly detrimental. 
        """
        #self.recurrent_network = self.initialise_weights(nn.Linear(self.input_size + self.hidden_size, self.hidden_size))
        if self.RNN_type == "rnn":
            self.recurrent_network = nn.RNN(input_size = self.input_size,
                                            hidden_size= self.hidden_size,
                                            num_layers = 1,
                                            nonlinearity = self.nonlinearity_name,
                                            batch_first = True)
        elif self.RNN_type == "lstm":
            self.recurrent_network = nn.LSTM(input_size = self.input_size,
                                             hidden_size= self.hidden_size,
                                             num_layers = 1,
                                             batch_first = True)
        else:
            raise ValueError("RNN type {} not recognised". format(self.RNN_type))

    def forward(self, input, hidden, c_0 = None) -> torch.Tensor:
        """
        Pass the inputs and the hidden through self.recurrent_network and applies the nonlinearity (typically tanh).
        """
        if self.RNN_type == "rnn":
            output, hidden  = self.recurrent_network(input, hidden) #note: nonlinearity inside the recurrent-network definition (how sweet?)
            # Output contains all hidden states. Shape: seq_len, batch, num_directions * hidden_size
            # hidden is the final hidden state.
            return output, hidden
        else:
            output, (hidden, c_state) = self.recurrent_network(input, (hidden, c_0)) #note: nonlinearity inside the recurrent-network definition (how sweet?)
            # Output contains all hidden states. Shape: seq_len, batch, num_directions * hidden_size
            # hidden is the final hidden state.
            return output, (hidden, c_state)

    def save_model(self, path) -> None:
        torch.save(self.state_dict(), os.path.join(path, 'saved_RNN_weights.pt'))
    
    def load_model(self, path) -> None:
        """
        Load saved weights into model.
        """
        saved_weights = torch.load(path)
        self.load_state_dict(saved_weights)
