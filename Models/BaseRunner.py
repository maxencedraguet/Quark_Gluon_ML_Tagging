#############################################################################
#
# BasicRunner.py
#
# Mother class of runner. Aimed at interacting with the main and a ML network
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

from DataLoaders import DataLoader_Set1, DataLoader_Set2, DataLoader_Set4

class _BaseRunner(ABC):
    def __init__(self, config: Dict) -> None:
        self.extract_parameters(config)
        self.setup_dataset(config)
    @abstractmethod
    def extract_parameters(self, config: Dict) -> None:
        raise NotImplementedError("Base class method")

    def checkpoint_df(self, step: int) -> None:
        raise NotImplementedError("Base class method")

    def setup_dataset(self, config: Dict)->None:
        if self.dataset == "Set1":
            self.dataloader = DataLoader_Set1(config)
        if self.dataset == "Set2":
            self.dataloader = DataLoader_Set2(config)
        self.data = self.dataloader.load_separate_data()

    def run(self):
        self.train()
        self.test()

    def train(self):
        raise NotImplementedError("Base class method")

    def test(self):
        raise NotImplementedError("Base class method")
    
    def assess(self):
        raise NotImplementedError("Base class method")
