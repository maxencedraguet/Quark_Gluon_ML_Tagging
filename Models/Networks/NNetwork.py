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
import warnings as warning

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseNetwork import _BaseNetwork

class NeuralNetwork(_BaseNetwork):
    
    def __init__(self, source, config: Dict):
        self.source = source
        self.extract_parameters(source, config)
        _BaseNetwork.__init__(self, config=config)
        # Methods inherited from mother class:
        self.nonlinearity = self.identify_nonlinfunc(self.nonlinearity_name)
        self.end_nonlinearity = self.identify_nonlinfunc(self.end_nonlinearity_name)
        self.construct_layers()
    
    def extract_parameters(self, source, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        dimensions should be a list of successive size.
        """
        if self.source == "NN_Model":
            self.dimensions = config.get(["NN_Model", "NeuralNet", "input_dimensions"])
            self.initialisation = config.get(["NN_Model", "initialisation"])
            self.proba_dropout_first_layer = config.get(["NN_Model", "dropout_proba"])
        
            self.nonlinearity_name = config.get(["NN_Model", "NeuralNet", "nonlinearity"])
            self.end_nonlinearity_name = config.get(["NN_Model", "NeuralNet", "end_nonlinearity"])
        
        elif self.source == "RecurrentInit":
            self.dimensions = config.get(["Junipr_Model", "Structure", "Recurrent", "Init", "input_dimensions"])
            self.initialisation = config.get(["Junipr_Model", "Structure", "Recurrent", "Init", "initialisation"])
            self.proba_dropout_first_layer = 0
            
            self.nonlinearity_name = config.get(["Junipr_Model", "Structure", "Recurrent", "Init", "nonlinearity"])
            self.end_nonlinearity_name = config.get(["Junipr_Model", "Structure", "Recurrent", "Init", "end_nonlinearity"])
            control_hidden_dimensions = config.get(["Junipr_Model", "Structure", "Recurrent", "hidden_dimensions"])
            if control_hidden_dimensions != self.dimensions[-1]:
                warning.warn("Output dimension of Recurrent Initial network = {} and does not agree with hidden dimension given {}.\nEnforcing hidden dimension upon network.".format(self.dimensions[-1], control_hidden_dimensions))
                self.dimensions[-1] = control_hidden_dimensions
                    
        elif self.source in ["JuniprEnd", "JuniprMother", "JuniprBranch", "JuniprBranchZ", "JuniprBranchT", "JuniprBranchD", "JuniprBranchP"]:
            # Define the gatherer
            def typical_NN_loader_JUNIPR(branch_name):
                """
                to retrieve the input associated to the branch being developped
                """
                self.dimensions = config.get(["Junipr_Model", "Structure", branch_name, "input_dimensions"])
                self.initialisation = config.get(["Junipr_Model", "Structure", branch_name,"initialisation"])
                self.proba_dropout_first_layer = 0
                
                self.nonlinearity_name = config.get(["Junipr_Model", "Structure", branch_name, "nonlinearity"])
                self.end_nonlinearity_name = config.get(["Junipr_Model", "Structure", branch_name, "end_nonlinearity"])
            
                # What follow is some dimensions enforcing with warning raised. First get the common info (network must agree with this)
                control_hidden_dimensions = config.get(["Junipr_Model", "Structure", "Recurrent", "hidden_dimensions"])
                control_padding_size = config.get(["Junipr_Model", "Junipr_Dataset", "padding_size"])
                control_granularity = config.get(["Junipr_Model", "Junipr_Dataset", "granularity"])
                
                if branch_name in ["JuniprEnd", "JuniprMother"]:
                    if control_hidden_dimensions != self.dimensions[0]:
                        warning.warn("Input dimension of {} network = {} and does not agree with hidden dimension given {}.\nEnforcing hidden dimension upon network.".format(branch_name, self.dimensions[0], control_hidden_dimensions))
                        self.dimensions[0] = control_hidden_dimensions
            
                if branch_name == "JuniprMother":
                    if control_padding_size != self.dimensions[-1]:
                        warning.warn("Output dimension of {} network = {} and does not agree with padding dimensions given {}.\nEnforcing padding dimensions upon network.".format(branch_name, self.dimensions[-1], control_padding_size))
                        self.dimensions[-1] = control_padding_size
                
                if branch_name == "JuniprBranch":
                    if ((control_hidden_dimensions + 4) != self.dimensions[0]):
                        warning.warn("Output dimension of {} network = {} and does not agree with granularity*4 dimension given {}.\nEnforcing padding dimensions upon network.".format(branch_name, self.dimensions[0], (control_hidden_dimensions + 4)))
                        self.dimensions[0] = (control_hidden_dimensions + 4)
                    if control_granularity * 4 != self.dimensions[-1]:
                        warning.warn("Output dimension of {} network = {} and does not agree with granularity*4 dimension given {}.\nEnforcing padding dimensions upon network.".format(branch_name, self.dimensions[-1], control_granularity * 4))
                        self.dimensions[-1] = control_granularity * 4
                
                if branch_name == ["JuniprBranchZ", "JuniprBranchT", "JuniprBranchD", "JuniprBranchP"]:
                    if control_granularity != self.dimensions[-1]:
                        warning.warn("Output dimension of {} network = {} and does not agree with granularity dimension given {}.\nEnforcing padding dimensions upon network.".format(branch_name, self.dimensions[-1], control_granularity))
                        self.dimensions[-1] = control_granularity
                
                if branch_name == "JuniprBranchz":
                    if ((control_hidden_dimensions + 4) != self.dimensions[0]):
                        warning.warn("Output dimension of {} network = {} and does not agree with granularity dimension given {}.\nEnforcing padding dimensions upon network.".format(branch_name, self.dimensions[0], (control_hidden_dimensions + 4)))
                        self.dimensions[0] = (control_hidden_dimensions + 4)
                        
                if branch_name == "JuniprBranchT":
                    if ((control_hidden_dimensions + 1 + 4) != self.dimensions[0]):
                        warning.warn("Output dimension of {} network = {} and does not agree with granularity dimension given {}.\nEnforcing padding dimensions upon network.".format(branch_name, self.dimensions[0], (control_hidden_dimensions + 1+ 4)))
                        self.dimensions[0] = (control_hidden_dimensions + 1 + 4)
                
                if branch_name == "JuniprBranchD":
                    if ((control_hidden_dimensions + 2 + 4) != self.dimensions[0]):
                        warning.warn("Output dimension of {} network = {} and does not agree with granularity dimension given {}.\nEnforcing padding dimensions upon network.".format(branch_name, self.dimensions[0], (control_hidden_dimensions + 2+ 4)))
                        self.dimensions[0] = (control_hidden_dimensions + 2 + 4)

                if branch_name == "JuniprBranchP":
                    if ((control_hidden_dimensions + 3 + 4)!= self.dimensions[0]):
                        warning.warn("Output dimension of {} network = {} and does not agree with granularity dimension given {}.\nEnforcing padding dimensions upon network.".format(branch_name, self.dimensions[0], (control_hidden_dimensions + 3+ 4)))
                        self.dimensions[0] = (control_hidden_dimensions + 3 + 4)
                    
            # end definition
            typical_NN_loader_JUNIPR(self.source)

        else:
            raise ValueError("Invalid source name {}".format(self.source))
                
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
