#############################################################################
#
# BaseNetwork.py
#
# Mother class of networks.
#
# Author -- Maxence Draguet (19/05/2020)
#
#############################################################################

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class _BaseNetwork(nn.Module, ABC):
    def __init__(self, config: Dict) -> None:
        super(_BaseNetwork, self).__init__()
        #self.nonlinearity = self.identify_nonlinfunc(self.nonlinearity_name)
        #self.end_nonlinearity = self.identify_nonlinfunc(self.end_nonlinearity_name)
        #self.construct_layers()
    
    def identify_nonlinfunc(self, name):
        if name == 'relu':
            name = F.relu
        elif name == 'elu':
            name = F.elu
        elif name == 'sigmoid':
            name = torch.sigmoid
        elif name == 'tanh':
            name = torch.tanh
        elif name == 'identity':
            name = nn.Identity()
        else:
            raise ValueError("Invalid nonlinearity name")
        return name

    def initialise_weights(self, layer):
        """
        Initialise the weights of the neural net.
        """
        if self.initialisation == "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        elif self.initialisation == "xavier_normal":
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        else:
            raise ValueError("Initialisation {} not recognised".format(self.initialisation))
        return layer

    @abstractmethod
    def extract_parameters(self, config: Dict) -> None:
        raise NotImplementedError("Base class method")


    def forward(self):
        raise NotImplementedError("Base class method")

