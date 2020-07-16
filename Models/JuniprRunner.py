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
from .Networks import JuniprNetwork
from Utils import write_ROC_info, plot_confusion_matrix, ROC_curve_plotter_from_values

class JuniprRunner(_BaseRunner):
    def __init__(self, config: Dict):
        self.extract_parameters(config)
        self.setup_Model(config)
        self.setup_optimiser(config)
        self.setup_dataloader(config)
        self.writer = SummaryWriter(self.result_path) # A tensorboard writer
        self.train()

    def extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        """
        self.verbose = True
        
        self.experiment_timestamp = config.get("experiment_timestamp")
        self.absolute_data_path = config.get(["absolute_data_path"])
        self.result_path = config.get(["log_path"])
        os.makedirs(self.result_path, exist_ok=True)
        self.dataset = config.get(["dataset"])
        self.save_model_bool = config.get(["save_model"])
        self.seed = config.get(["seed"])
        
        self.lr = config.get(["Junipr_Model", "lr"])
        #self.lr_scheduler = config.get(["Junipr_Model", "lr_scheduler"])
        self.num_epochs = config.get(["Junipr_Model", "epoch"])
        self.batch_size = config.get(["Junipr_Model", "batch_size"])
        self.test_frequency = config.get(["Junipr_Model", "test_frequency"])
        self.optimiser_type = config.get(["Junipr_Model", "optimiser", "type"])
        self.optimiser_params = config.get(["Junipr_Model", "optimiser", "params"])
        

        self.branch_treatment = config.get(["Junipr_Model", "Structure", "branch_structure"])    #whether to represent branch branch in 1 or 4 networks
        self.padding_size = config.get(["Junipr_Model", "Junipr_Dataset", "padding_size"])
        self.padding_value= config.get(["Junipr_Model", "Junipr_Dataset", "padding_value"])
            
    def setup_Model(self, config: Dict):
        """
        To set up the different components of the JUNIPR model (as well as loss functions)
        """
        self.JUNIPR_model = JuniprNetwork(config=config)
        self.setup_loss()
    
    def setup_dataloader(self, config: Dict)->None:
        """
        Set up the dataloader for PyTorch execution.
        Note that the batched samples are dictionnaries here.
        """
        if self.dataset == "Set4":
            self.dataloader = DataLoader_Set4(config)
        else:
            raise ValueError("Dataset {} not appropriate for JUNIPR model". format(self.dataset))
        self.train_dataset, self.test_dataset = self.dataloader.load_separate_data()
    
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader  = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False)

    def setup_optimiser(self, config: Dict):
        if self.optimiser_type == "adam":
            beta_1 = self.optimiser_params[0]
            beta_2 = self.optimiser_params[1]
            epsilon = self.optimiser_params[2]
            self.optimiser = torch.optim.Adam(self.JUNIPR_model.parameters(), lr=self.lr,
                                              betas=(beta_1, beta_2),eps=epsilon) #weight_decay= self.weight_decay
        else:
            raise ValueError("Optimiser {} not recognised". format(self.optimiser_type))
                
        """
        # This needs work: it is to adapt the learning rate in training
        self.lr_scheduler_bool = False
        if self.lr_scheduler == "own":
            self.lr_scheduler_bool = True
            self.lr_schedule = {"" }
        """
        """
        # This needs work: it is to adapt the batchsize in training
        self.batchsize_scheduler_bool = False
        if self.batchsize_scheduler == "junipr_paper":
            self.batchsize_scheduler_bool = True
            # keys indicate the epoch at which the value is to be set as batchsize.
            self.batchsize_schedule = {"1": 10, "2": 100, "7": 1000, "17": 2000 }
        """
                
    def setup_loss(self):
        """
        Gives the class a dictionnary of loss functions (for each of the three/six modules) in self.loss_modules
        
        In Junipr, the loss is in fact the probability of the data (the likelihood) and needs to be maximised.
        This is obviously equivalent to minimising minus the probability (or the be more precise, the log probability).
        
        Note:
            - three modules if the branch one is a single step
            - six modules if the branch one is in fact 4 modules (seperating each components)
        """
        self.loss_modules = dict()
    
        # For P_end: a binary categorical cross-entropy
        # Warning on BCEWithLogitsLoss !
        # binary cross entropy loss with logits includes a sigmoid transform! Do not add it to the network!
        self.loss_p_end = nn.BCEWithLogitsLoss(reduction = 'none')
        self.loss_modules["p_end"] = self.loss_p_end
        
        # For P_mother: a categorical cross-entropy
        # Warning on CrossEntropyLoss !
        # It combines a log(softmax) with NLL_loss so do not add softmax to the mother network !
        self.loss_p_mother  = nn.CrossEntropyLoss(reduction = 'none')
        self.loss_modules["p_mother"] = self.loss_p_mother
        
        # For P_branch: 4 categorical cross-entropy functions
        # The inputs of these losses will differ in single or mutliple scenarios, the losses themselves not
        self.loss_p_branch_z  = nn.CrossEntropyLoss(reduction = 'none')
        self.loss_modules["p_branch_z"] = self.loss_p_branch_z
        self.loss_p_branch_t  = nn.CrossEntropyLoss(reduction = 'none')
        self.loss_modules["p_branch_t"] = self.loss_p_branch_t
        self.loss_p_branch_p  = nn.CrossEntropyLoss(reduction = 'none')
        self.loss_modules["p_branch_p"] = self.loss_p_branch_p
        self.loss_p_branch_d  = nn.CrossEntropyLoss(reduction = 'none')
        self.loss_modules["p_branch_d"] = self.loss_p_branch_d
    
    def compute_losses(self, batch_input, batch_output):
        """
        To compute the loss in our batch-leaning environment: the sequences are flatten into an array.
        
        Each sequence in a batch will in fact be flatten into a single large array (this is therefore flattening
        accross batched samples: no problem for computing losses but this demand a batch-size of 1 for analysing an event.
        
        batch_input, batch_output are dictionnaries with the entries required.
        
        Note that the ignore_index specification from the nn.CrossEntropyLoss() is not sufficient here: padded values are passed through the networks.
        """
        batch_loss = 0
        # Input info needed
        input_n_branchings = batch_input["n_branchings"]
        #input_ending      = batch_input["ending"]  #Not necessary: remove from dataset: n_branchings is enough
        input_branching   = batch_input["branching"]
        input_mother_id   = batch_input["mother_id_energy"]
        # Output info needed
        output_end      = batch_output["output_end"]
        output_very_end = batch_output["output_very_end"]
        output_mother   = batch_output["output_mother"]
        output_branch_z = batch_output["output_branch_z"]
        output_branch_t = batch_output["output_branch_t"]
        output_branch_p = batch_output["output_branch_p"]
        output_branch_d = batch_output["output_branch_d"]
        
        # Cast input so that the padded size is the same the output one (normally it's the size of the longest item)
        #print(" In junipr runner loss module compute, input_ending : {}".format(input_ending[1, :]))
        #input_ending    = input_ending[:, :output_mother.size()[1]]
        input_branching = input_branching[:, :output_mother.size()[1], :]
        input_mother_id = input_mother_id[:, :output_mother.size()[1]]
   
        """
        # To check value of outputs
        print("In junipr runner loss module, output_end : {}".format(output_end))
        print("In junipr runner loss module, output_very_end : {}".format(output_very_end))
        print("In junipr runner loss module, output_mother : {}".format(output_mother))
        print("In junipr runner loss module, output_branch_z : {}".format(output_branch_z))
        print("In junipr runner loss module, output_branch_t : {}".format(output_branch_t))
        print("In junipr runner loss module, output_branch_p : {}".format(output_branch_p))
        print("In junipr runner loss module, output_branch_d : {}".format(output_branch_d))
        """
        
        element_input_very_ending = torch.FloatTensor([1])
        for batch_element in range(input_n_branchings.size()[0]):
            element_n_branching       = input_n_branchings[batch_element].item()
            # consider the next item by restricting them to the real size of the sequence: element_n_branching.
            element_input_branching   = input_branching[batch_element, :element_n_branching, :]
            element_input_mother_id   = input_mother_id[batch_element, :element_n_branching]
            element_output_end        = output_end[batch_element, :element_n_branching, :]
            element_output_very_end   = output_very_end[0, batch_element, :] #there is a first component of size 1 that is due to the RNN structure (fixed to 1 here)
            element_output_mother     = output_mother[batch_element, :element_n_branching, :element_n_branching]
            element_output_branch_z   = output_branch_z[batch_element, :element_n_branching, :]
            element_output_branch_t   = output_branch_t[batch_element, :element_n_branching, :]
            element_output_branch_p   = output_branch_p[batch_element, :element_n_branching, :]
            element_output_branch_d   = output_branch_d[batch_element, :element_n_branching, :]
            
            # tensor representing ending value (0 go on, 1 means end but the 1 is contained in element_input_very_ending defined above).
            element_input_ending = torch.FloatTensor(np.zeros((element_n_branching, 1)))
            
            p_mother_output_loss   = self.loss_modules["p_mother"](element_output_mother, element_input_mother_id.long())
            p_end_output_loss      = self.loss_modules["p_end"](element_output_end[:, 0], element_input_ending[:, 0])
            p_very_end_output_loss = self.loss_modules["p_end"](element_output_very_end, element_input_very_ending)
            p_branch_z_output_loss = self.loss_modules["p_branch_z"](element_output_branch_z, element_input_branching[:, 0].long())
            p_branch_t_output_loss = self.loss_modules["p_branch_t"](element_output_branch_t, element_input_branching[:, 1].long())
            p_branch_p_output_loss = self.loss_modules["p_branch_p"](element_output_branch_p, element_input_branching[:, 2].long())
            p_branch_d_output_loss = self.loss_modules["p_branch_d"](element_output_branch_d, element_input_branching[:, 3].long())
            
            list_losses = [p_mother_output_loss, p_end_output_loss, p_very_end_output_loss, p_branch_z_output_loss,
                           p_branch_t_output_loss, p_branch_p_output_loss, p_branch_d_output_loss]
            losses_combined = torch.cat(list_losses, dim = 0).sum(dim=0) / element_n_branching
            batch_loss += losses_combined
            #print("In junipr runner loss module compute, global loss : {}".format(losses_combined))
        return batch_loss
    
    def train(self):
        """
        The training loop. Note that manipulation of the input/output into the losses is very particular here.
        """
        # Set model to train mode
        self.JUNIPR_model.train()
        step_count = 0
        
        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0
            epoch_acc  = 0
            batch_count = 0
            # If a learning rate schedule is used:
            """
            if self.lr_scheduler:
                if (epoch == int((self.num_epochs + 1)/3) or epoch == int((self.num_epochs + 1)/3)*2):
                    self.lr *= 0.5
                        print("Learning Rate schedule: ", self.lr)
                        # Update the learning rate parameter of the optimiser.
                        for param_group in self.optimiser.param_groups:
                            param_group['lr'] = self.lr
            """
            for batch_input in self.train_dataloader:
                step_count += 1
                batch_count += 1
                input_n_branchings = batch_input["n_branchings"]
                size_batch = input_n_branchings.size()[0]
                
                self.JUNIPR_model.RNN_needs_initialisation = True
                
                batch_output = self.JUNIPR_model(batch_input)
                
                batch_loss = self.compute_losses(batch_input, batch_output) / size_batch
                epoch_loss += batch_loss
                
                self.optimiser.zero_grad()
                batch_loss.backward()
                self.optimiser.step()

                # Report result to TensorBoard
                self.writer.add_scalar("training_loss", float(batch_loss), step_count)
                print("In training loop, batch loss : {}".format(batch_loss))
            print("\n#######################################################\n")
            print("Epoch {} loss : {}".format(epoch, epoch_loss/batch_count))
            print("\n#######################################################\n")

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



