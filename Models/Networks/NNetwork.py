#############################################################################
#
# NNRunner.py
#
# A neural network runner using PyTorch
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
    
    def __init__(self, config: Dict):
        self.extract_parameters(config)
        _BaseNetwork.__init__(self, config=config)
        self.construct_layers()

    
    def extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        dimensions should be a list of successive size.
        """
        self.dimensions = config.get(["NN_Model", "NeuralNet", "input_dimensions"])
        self.initialisation = config.get(["NN_Model", "initialisation"])
        self.proba_dropout_first_layer = config.get(["NN_Model", "dropout_proba"])

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
