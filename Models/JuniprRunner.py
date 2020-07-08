#############################################################################
#
# JuniprRunner.py
#
# A JUNIPR runner using PyTorch to implement the JUNIPR model
#
# Author -- Maxence Draguet (06/07/2020)
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
import torch.nn.functional as F
from sklearn import metrics

from tensorboardX import SummaryWriter

from DataLoaders import DataLoader_Set4
from .BaseRunner import _BaseRunner
from .Networks import NeuralNetwork#, RecurrentNetwork
from Utils import write_ROC_info, plot_confusion_matrix, ROC_curve_plotter_from_values

class JuniprRunner(_BaseRunner):
    def __init__(self, config: Dict):
        self.extract_parameters(config)
        self.setup_Model(config)
        self.setup_optimiser(config)
        self.setup_dataloader(config)
        self.writer = SummaryWriter(self.result_path) # A tensorboard writer
        self.run()

    def extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        """
        self.verbose = False
        
        self.experiment_timestamp = config.get("experiment_timestamp")
        self.absolute_data_path = config.get(["absolute_data_path"])
        self.result_path = config.get(["log_path"])
        os.makedirs(self.result_path, exist_ok=True)
        self.dataset = config.get(["dataset"])
        self.save_model_bool = config.get(["save_model"])
        self.seed = config.get(["seed"])
        
        # dimensions should be a list of successive layers.
        self.lr = config.get(["Junipr_Model", "lr"])
        self.lr_scheduler = config.get(["Junipr_Model", "lr_scheduler"])
        self.num_epochs = config.get(["Junipr_Model", "epoch"])
        self.test_frequency = config.get(["Junipr_Model", "test_frequency"])
        self.optimiser_type = config.get(["Junipr_Model", "optimiser", "type"])
        self.optimiser_params = config.get(["Junipr_Model", "optimiser", "params"])
        self.weight_decay = config.get(["Junipr_Model", "optimiser", "weight_decay"])
        self.batch_size = config.get(["Junipr_Model", "batch_size"])
        self.loss_function = config.get(["Junipr_Model", "loss_function"])

    def setup_Model(self, config: Dict):
        """
        To set up the different components of the JUNIPR model
        """
        # Let's first set up the RNN.
        
        # We can then set up the various MLP taking the hidden state of the RNN to produce
        #   - P_end
        
        #   - P_mother
        
        #   - P_branch
    
    def setup_dataloader(self, config: Dict)->None:
        """
        Set up the dataloader for PyTorch execution
        
        Has to convert the difficult input into tensor
        """
        if self.dataset == "Set4":
            self.dataloader = DataLoader_Set4(config)
        else:
            raise ValueError("Dataset {} not appropriate for JUNIPR model". format(self.dataset))
        self.train_dataset, self.test_dataset = self.dataloader.load_separate_data()
    
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader  = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False)


    def setup_optimiser(self, config: Dict):
        pass

    def setup_loss(self, config: Dict):
        pass

    def run(self):
        if self.verbose:
            print("Train\n")
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                print(" Mother and daugther mom \n",i_batch, sample_batched['mother_momenta'].size(), sample_batched['daughter_momenta'].size())
                print(" Label and multiplicity \n",i_batch, sample_batched['label'].size(), sample_batched['multiplicity'].size())
                print(" n_branching and seed \n",i_batch, sample_batched['n_branchings'].size(), sample_batched['seed_momentum'].size())
                print(" branching \n",i_batch, sample_batched['branching'].size())
                #print(sample_batched['branching'][0][0])
                print(" mother_id_energy \n",i_batch, sample_batched['mother_id_energy'].size())
                print(" ending \n",i_batch, sample_batched['ending'].size())
                
                print(" Mother and daugther mom \n",i_batch, type(sample_batched['mother_momenta']), type(sample_batched['daughter_momenta']))
                print(" Label and multiplicity \n",i_batch, type(sample_batched['label']), type(sample_batched['multiplicity']))
                print(" n_branching and seed \n",i_batch, type(sample_batched['n_branchings']), type(sample_batched['seed_momentum']))
                print(" branching \n",i_batch, type(sample_batched['branching']))
                #print(sample_batched['branching'][0][0])
                print(" mother_id_energy \n",i_batch, type(sample_batched['mother_id_energy']))
                print(" ending \n",i_batch, type(sample_batched['ending']))
            print("\nTest\n")
            for i_batch, sample_batched in enumerate(self.test_dataloader):
                print(" Mother and daugther mom \n",i_batch, sample_batched['mother_momenta'].size(), sample_batched['daughter_momenta'].size())
                print(" Label and multiplicity \n",i_batch, sample_batched['label'].size(), sample_batched['multiplicity'].size())
                print(" n_branching and seed \n",i_batch, sample_batched['n_branchings'].size(), sample_batched['seed_momentum'].size())
                print(" branching \n",i_batch, sample_batched['branching'].size())
                #print(sample_batched['branching'][0][0])
                print(" mother_id_energy \n",i_batch, sample_batched['mother_id_energy'].size())
                print(" ending \n",i_batch, sample_batched['ending'].size())
            
                print(" Mother and daugther mom \n",i_batch, type(sample_batched['mother_momenta']), type(sample_batched['daughter_momenta']))
                print(" Label and multiplicity \n",i_batch, type(sample_batched['label']), type(sample_batched['multiplicity']))
                print(" n_branching and seed \n",i_batch, type(sample_batched['n_branchings']), type(sample_batched['seed_momentum']))
                print(" branching \n",i_batch, type(sample_batched['branching']))
                #print(sample_batched['branching'][0][0])
                print(" mother_id_energy \n",i_batch, type(sample_batched['mother_id_energy']))
                print(" ending \n",i_batch, type(sample_batched['ending']))



