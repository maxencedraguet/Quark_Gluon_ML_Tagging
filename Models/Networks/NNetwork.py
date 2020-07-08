#############################################################################
#
# NNetwork.py
#
# A neural network using PyTorch
#
# Author -- Maxence Draguet (1/06/2020)
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

class NeuralNetwork(_BaseNetwork):
    
    def __init__(self, source: String, config: Dict):
        self.source = source
        self.extract_parameters(config)
        _BaseNetwork.__init__(self, config=config)
        
        # Methods inherited from mother class:
        self.nonlinearity = self.identify_nonlinfunc(self.nonlinearity_name)
        self.end_nonlinearity = self.identify_nonlinfunc(self.end_nonlinearity_name)
        self.construct_layers()

    
    def extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        dimensions should be a list of successive size.
        """
        if self.source = "NN_Model":
            self.dimensions = config.get(["NN_Model", "NeuralNet", "input_dimensions"])
            self.initialisation = config.get(["NN_Model", "initialisation"])
            self.proba_dropout_first_layer = config.get(["NN_Model", "dropout_proba"])
        
            self.nonlinearity_name = config.get(["NN_Model", "NeuralNet", "nonlinearity"])
            self.end_nonlinearity_name = config.get(["NN_Model", "NeuralNet", "end_nonlinearity"])
        
        elif self.source == "RecurrentInit":
            self.dimensions = config.get(["Junipr_Model", "Structure", "Recurrent", "Init", "input_dimensions"])
            self.initialisation = config.get(["Junipr_Model", "Structure", "Recurrent", "Init", "initialisation"])
            self.proba_dropout_first_layer = config.get(["Junipr_Model", "Structure", "Recurrent", "Init",])
            
            self.nonlinearity_name = config.get(["Junipr_Model", "Structure", "Recurrent", "Init",])
            self.end_nonlinearity_name = config.get(["Junipr_Model", "Structure", "Recurrent", "Init", "end_nonlinearity"])
        elif self.source == "JuniprEnd":
            pass
        elif self.source == "JuniprMother":
            pass
        elif self.source == "JuniprBranchZ":
            pass
        elif self.source == "JuniprBranchT":
            pass
        elif self.source == "JuniprBranchD":
            pass
        elif self.source == "JuniprBranchP":
            pass
        else:
            raise ValueError("Invalid source name")
                
    def get_last_non_linearity(self):
        return self.end_nonlinearity_name

    def construct_layers(self) -> None:
        """
        Constructs the architecture
        """
        self.layers = nn.ModuleList([])
        for h in range(len(self.dimensions)-1):
            #print("Layer {}: from {} to {}".format(h, self.dimensions[h], self.dimensions[h + 1]))
            self.layers.append(self.initialise_weights(nn.Linear(self.dimensions[h], self.dimensions[h + 1])))

        self.dropout_first_layer = nn.Dropout(p= self.proba_dropout_first_layer)

    def forward(self, x) -> torch.Tensor:
        """
        Pass the inputs through the layers.
        """
        #print("Start Foward {} and shape {}".format(x, x.shape))
        for count, layer in enumerate(self.layers[:-1]):
            x = self.nonlinearity(layer(x))
            if count == 0:
                x = self.dropout_first_layer(x)
        
        #print("Layer {}: {} and shape {}".format(count, x, x.shape))
        return self.end_nonlinearity(self.layers[-1](x))

    def save_model(self, path) -> None:
        torch.save(self.state_dict(), os.path.join(path, 'saved_NN_weights.pt'))
    
    def load_model(self, path) -> None:
        """
        Load saved weights into model.
        """
        saved_weights = torch.load(path)
        self.load_state_dict(saved_weights)
