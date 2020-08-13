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
import io
import yaml
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from joblib import dump, load

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from sklearn import metrics

from tensorboardX import SummaryWriter
import tensorflow as tf

from DataLoaders import DataLoader_Set4
from .BaseRunner import _BaseRunner
from .Networks import JuniprNetwork
from Utils import tree_plotter, draw_tree, write_ROC_info, plot_confusion_matrix, ROC_curve_plotter_from_values, write_to_file
from Utils import MainParameters, JUNIPR_distributions_plot

class JuniprRunner(_BaseRunner):
    def __init__(self, config: Dict):
        self.extract_parameters(config)
        self.setup_Model(config)
        self.setup_optimiser()
        self.setup_dataloader(config)
        if self.binary_junipr:
            print("Running a binary version of JUNIPR")
            # training a binary version of junipr. This will require loading two pre-trained models and saving these double models.
            self.previously_trained_number_epoch = config.get(["Junipr_Model", "binary_runner", "pre_trained_epochs"])
            self.previously_trained_number_steps = config.get(["Junipr_Model", "binary_runner", "previously_trained_number_steps"])
            
            quark_model_path  = config.get(["Junipr_Model", "binary_runner", "quark_model_path"])
            quark_config_path = config.get(["Junipr_Model", "binary_runner", "quark_config_path"])
            
            gluon_model_path  = config.get(["Junipr_Model", "binary_runner", "gluon_model_path"])
            gluon_config_path = config.get(["Junipr_Model", "binary_runner", "gluon_config_path"])
            
            print("The quark model comes from {}".format(quark_model_path))
            print("The gluon model comes from {}".format(gluon_model_path))
            with open(quark_config_path, 'r') as yaml_file_quark:
                quark_loaded_parameters = yaml.load(yaml_file_quark, yaml.SafeLoader)
            with open(gluon_config_path, 'r') as yaml_file_gluon:
                gluon_loaded_parameters = yaml.load(yaml_file_gluon, yaml.SafeLoader)
        
            self.quark_additional_parameters = MainParameters(quark_loaded_parameters)
            self.gluon_additional_parameters = MainParameters(gluon_loaded_parameters)
            
            self.JUNIPR_quark_model = JuniprNetwork(config=self.quark_additional_parameters)
            self.JUNIPR_gluon_model = JuniprNetwork(config=self.gluon_additional_parameters)
            self.JUNIPR_quark_model.load_model(quark_model_path)
            self.JUNIPR_gluon_model.load_model(gluon_model_path)
            if self.train_bool:
                print("Training the binary Junipr model")   # remark that in this case there must be a loading step
                self.writer = SummaryWriter(self.result_path) # A tensorboard writer
                self.run(config, train_bool = True, save_model_bool = self.save_model_bool , binary_junipr_bool = True)
            if self.assess_bool:
                print("Assessing the binary Junipr model")
                self.run(config, assess_bool = True, print_jets_bool = True, binary_junipr_bool = True)
        else:
            if self.load_bool:
                print("Loading a Junipr model")
                self.run(config, load_bool = True)
            if self.train_bool:
                print("Training a Junipr model")    # could also be retraining a loaded model
                self.writer = SummaryWriter(self.result_path) # A tensorboard writer
                self.run(config, train_bool = True, save_model_bool = self.save_model_bool)
            if self.assess_bool:
                print("Assessing the Junipr model")
                self.run(config, assess_bool = True, print_jets_bool = True)
            
        print("\nFinished running the JUNIPR Runner\n")
                  
    def extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        """
        self.verbose = True
        
        self.experiment_timestamp = config.get("experiment_timestamp")
        self.absolute_data_path = config.get(["absolute_data_path"])
        self.result_path = config.get(["log_path"])
        self.logger_data_bool = config.get(["logger_data"])
        os.makedirs(self.result_path, exist_ok=True)
        self.dataset = config.get(["dataset"])
        self.seed = config.get(["seed"])
        
        self.train_bool  = config.get(["Junipr_Model", "train"])   # If true, will train a model (either defined here or loaded
        self.assess_bool = config.get(["Junipr_Model", "assess"])  # If true, will assess the model
        self.load_bool   = config.get(["Junipr_Model", "load"])    # If true, will load a model and use this one. If false, will create a new model
        self.save_model_bool = config.get(["save_model"])          # if true, will save the model(s)
        
        self.binary_junipr = config.get(["Junipr_Model", "binary_runner_bool"]) # whether to run a binary version of junipr. Requires two pre-trained models.
        
        self.lr = config.get(["Junipr_Model", "lr"])
        self.lr_scheduler = config.get(["Junipr_Model", "lr_scheduler"])
        self.batch_scheduler = config.get(["Junipr_Model", "batch_scheduler"])
        self.num_epochs = config.get(["Junipr_Model", "epoch"])
        self.batch_size = config.get(["Junipr_Model", "batch_size"])
        self.test_frequency = config.get(["Junipr_Model", "test_frequency"])
        self.optimiser_type = config.get(["Junipr_Model", "optimiser", "type"])
        #self.optimiser_params = config.get(["Junipr_Model", "optimiser", "params"])

        self.branch_treatment = config.get(["Junipr_Model", "Structure", "branch_structure"])    # Whether to represent branch branch in 1 or 4 networks
        self.padding_size = config.get(["Junipr_Model", "Junipr_Dataset", "padding_size"])
        self.padding_value= config.get(["Junipr_Model", "Junipr_Dataset", "padding_value"])
        self.granularity= config.get(["Junipr_Model", "Junipr_Dataset", "granularity"])
        
        # Next parameter is used when further training a model (and modified in loading step of run), to restart at the last epoch trained (import for schedule).
        # The model will still be trained for the num_epochs given (it will be added to the current value so that the model will have globally trained
        # for previously_trained_number_epoch + self.num_epochs.
        self.previously_trained_number_epoch = 0
        self.previously_trained_number_steps = 0
        # Next parameter relates to binary runner: whether to use a final BCE with the truth label
        self.binary_BCE_loss = config.get(["Junipr_Model", "binary_runner", "end_BCE_bool"])

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
        
        How to select the number of workers ? A good discussion is available at: https://deeplizard.com/learn/video/kWVgvsejXsE
        """
        if self.dataset == "Set4":
            self.dataloader = DataLoader_Set4(config)
        else:
            raise ValueError("Dataset {} not appropriate for JUNIPR model". format(self.dataset))
        self.train_dataset, self.validation_dataset, self.test_dataset = self.dataloader.load_separate_data()

        bool_suffle_test = False
        if self.binary_junipr:
            # You need a large validation set (to choose your model) and a small one to check performance along training.
            # This here extracts a small one for the validation dataset. It's size is forced to be half the large one.
            validation_set_full_size = int(len(self.validation_dataset))
            val_size = int(validation_set_full_size / 2)
            rest_size = validation_set_full_size - val_size
            self.small_validation_dataset, _ = random_split(self.validation_dataset, [val_size, rest_size])
            self.small_validation_dataloader = torch.utils.data.DataLoader(self.small_validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 0)
            print("Enabled a small validation dataset of {} jets from the main validation dataset of {} jets".format(len(self.small_validation_dataset), len(self.validation_dataset)))
            if self.assess_bool:
                # Running a binary junipr and assessing it.
                # You have to shuffle the test in this case as otherwise you'll only draw the x first jets
                # which will be mostly quark if not only them.
                bool_suffle_test = True
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = 0)
        self.validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 0)
        print("SETTING UP DATALOADER, test size used for VALIDATION? ", self.batch_size)
        self.test_dataloader  = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=bool_suffle_test)

    def reset_dataloader(self, new_batchsize, epoch):
        """
        This is used to change the batchsize of the train dataloader (to change before iterating over it)
        """
        print("\n#######################################################\n")
        print("Epoch {} | New batch size : {}".format(epoch, new_batchsize) )
        print("\n#######################################################\n")
        self.batch_size = new_batchsize
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=new_batchsize, shuffle=True, num_workers = 0)

    def setup_optimiser(self, binary_case = False):
        if self.optimiser_type == "adam":
            """
            # Not specifying them at the moment
            beta_1 = self.optimiser_params[0]
            beta_2 = self.optimiser_params[1]
            epsilon = self.optimiser_params[2]
            """
            if binary_case:
                #self.optimiser = torch.optim.Adam(list(self.JUNIPR_quark_model.parameters()) + list(self.JUNIPR_gluon_model.parameters()), lr=self.lr)#,betas=(beta_1, beta_2),eps=epsilon)
                self.optimiser = torch.optim.Adam(list(self.JUNIPR_quark_model.parameters()) + list(self.JUNIPR_gluon_model.parameters()), lr=self.lr, amsgrad = True)
            else:
                self.optimiser = torch.optim.Adam(self.JUNIPR_model.parameters(), lr=self.lr)#,betas=(beta_1, beta_2),eps=epsilon) #weight_decay= self.weight_decay
        else:
            raise ValueError("Optimiser {} not recognised". format(self.optimiser_type))
                
        
        # Learning rate schedule
        self.lr_scheduler_bool = False
        if self.lr_scheduler == "5epochsJ":  # same as in main JUNIPR paper
            self.lr_scheduler_bool = True
            self.lr_schedule = {"1": 0.01, "6": 0.001, "11": 0.0001, "16": 0.001, "21": 0.0001, "26": 0.00001}
        elif self.lr_scheduler == "5epochsJL":  # same as in main JUNIPR paper but longer
            self.lr_scheduler_bool = True
            self.lr_schedule = {"1": 0.01, "6": 0.001, "11": 0.001, "16": 0.0001, "21": 0.0001, "26": 0.001, "31": 0.001, "36": 0.0001, "41": 0.00001, "46": 0.00001}
        elif self.lr_scheduler == "5epochsD":  # straight down schedule
            self.lr_scheduler_bool = True
            self.lr_schedule = {"1": 0.01, "6": 0.005, "11": 0.001, "16": 0.0005, "21": 0.0001, "26": 0.00001}
        elif self.lr_scheduler == "5epochsDL":  # longer schedule for 50 epochs
            self.lr_scheduler_bool = True
            self.lr_schedule = {"1": 0.01, "6": 0.005, "11": 0.001, "16": 0.001, "21": 0.001, "26": 0.001, "31": 0.0005, "36": 0.0005, "41": 0.00001, "46": 0.00001}
        elif self.lr_scheduler == "special_binary":  # particular schedule increase the learning rate a bit to match increase in batchsize. This seems motivated in the litterature
            self.lr_scheduler_bool = True
            self.lr_schedule = {"1": 0.001, "8": 0.005, "18": 0.01}


        # Batch size  schedule
        self.batchsize_scheduler_bool = False
        if self.batch_scheduler == "junipr_binary":
            self.batchsize_scheduler_bool = True
            self.batchsize_schedule = {"1": 10, "2": 100, "7": 1000, "17": 2000 }
        elif self.batch_scheduler == "junipr_unary_LONG":
            self.batchsize_scheduler_bool = True
            self.batchsize_schedule = {"1": 10, "26": 100}

        elif self.batch_scheduler == "junipr_binary_DOUBLE":
            self.batchsize_scheduler_bool = True
            self.batchsize_schedule = {"1": 10, "3": 100, "13": 1000, "33": 2000 }
        
            
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
        
        if self.binary_junipr and self.binary_BCE_loss:
            # Again, the BCEWithLogitsLoss will perform the sigmoid itself
            self.discriminator  = nn.BCEWithLogitsLoss(reduction = 'mean')
            self.loss_modules["discriminator"] = self.discriminator

    def compute_batch_losses(self, batch_input, batch_output, test_bool = False, binary_target = False):
        """
        Computes the loss for unayr JUNIPR in a batch-oriented way. This loss is in fact - log(Probability) (log in base e).
        
        The arguments batch_input, batch_output are dictionnaries with the info required.
        
        If test_bool = True, it means you return the batch loss and a list of tuples (one for each event) with the loss of the event and the loss per node.
        This is only really necessary when plotting a junipr tree and should be avoided if possible (slow).
        
        If binary_target = True, it means you use the loss here for the binary junipr. It requires to return the batch loss as a tensor with the loss for each sample. It is slightly slower than the regular computation.
        """
        #start_unary_loss = time.process_time()
        batch_loss = 0
        # Input info needed
        input_n_branchings = batch_input["n_branchings"]
        input_branching    = batch_input["branching"]
        input_mother_id    = batch_input["mother_id_energy"]
        input_branch_mask  = batch_input["branching_mask"]
        #input_mother_mask = batch_input["mother_mask"]
        
        # Output info needed
        unshaped_output_end      = batch_output["output_end"]
        unshaped_output_very_end = batch_output["output_very_end"]
        unshaped_output_mother   = batch_output["output_mother"]
        unshaped_output_branch_z = batch_output["output_branch_z"]
        unshaped_output_branch_t = batch_output["output_branch_t"]
        unshaped_output_branch_p = batch_output["output_branch_p"]
        unshaped_output_branch_d = batch_output["output_branch_d"]
 
        batch_size = unshaped_output_mother.size()[0]
        recurrent_limit = unshaped_output_mother.size()[1]

        # Cast input so that the padded size is the same the output one (normally it's the size of the longest item)
        """
        sum_n_branchigns = input_n_branchings.sum()
        print("sum_n_branchigns: ", sum_n_branchigns.item())
        print("n_branchigns: ", input_n_branchings)
        """
        
        input_branching = input_branching[:, :recurrent_limit, :].detach()
        input_mother_id = input_mother_id[:, :recurrent_limit].detach()
        input_branch_mask = input_branch_mask[:, :recurrent_limit].detach()
        #input_mother_mask = input_mother_mask[:, :recurrent_limit, :recurrent_limit].detach()
        """
        print("#############################################")
        print("\n Listing start size \n")
        print("#############################################")
        print("Size of output_end: {}".format(          output_end.size()))
        print("Size of output_very_end: {}".format(     output_very_end.size()))
        print("Size of output_mother: {}".format(       output_mother.size()))
        print("Size of output_branch_z: {}".format(     output_branch_z.size()))
        print("Size of output_branch_t: {}".format(     output_branch_t.size()))
        print("Size of output_branch_p: {}".format(     output_branch_p.size()))
        print("Size of output_branch_d: {}".format(     output_branch_d.size()))
        print("Size of input_n_branchings: {}".format(  input_n_branchings.size()))
        print("Size of input_branching: {}".format(     input_branching.size()))
        print("Size of input_mother_id: {}".format(     input_mother_id.size()))
        print("Size of input_branch_mask: {}".format(   input_branch_mask.size()))
        """
        # You need to reshape things for the losses. Unroll each batch sample into a linear tensor, sample after sample
        output_very_end = unshaped_output_very_end[:, 0, 0]
        # [batch_element, :element_n_branching, :] you want [batch_element X element_n_branching, :]
        #output_end = torch.reshape(unshaped_output_end, (-1,))   #output_end.size()[2]
        output_end = unshaped_output_end.view(-1,)
        
        #output_mother[batch_element, :element_n_branching, :element_n_branching]
        #output_mother= torch.reshape(unshaped_output_mother, (-1, unshaped_output_mother.size()[2]))
        output_mother= unshaped_output_mother.view(-1, unshaped_output_mother.size()[2])
        
        #print("Before reshape: output_mother \n", output_mother)
        #output_mother= torch.reshape(output_mother, (-1,))
        #print("AFTER reshape: output_mother \n", output_mother)
        #output_branch_z[batch_element, :element_n_branching, :]
        #output_branch_t[batch_element, :element_n_branching, :]
        #output_branch_p[batch_element, :element_n_branching, :]
        #output_branch_d[batch_element, :element_n_branching, :]
        #output_branch_z = torch.reshape(unshaped_output_branch_z, (-1, unshaped_output_branch_z.size()[2]))
        #output_branch_t = torch.reshape(unshaped_output_branch_t, (-1, unshaped_output_branch_t.size()[2]))
        #output_branch_p = torch.reshape(unshaped_output_branch_p, (-1, unshaped_output_branch_p.size()[2]))
        #output_branch_d = torch.reshape(unshaped_output_branch_d, (-1, unshaped_output_branch_d.size()[2]))
        
        output_branch_z = unshaped_output_branch_z.view(-1, unshaped_output_branch_z.size()[2])
        output_branch_t = unshaped_output_branch_t.view(-1, unshaped_output_branch_t.size()[2])
        output_branch_p = unshaped_output_branch_p.view(-1, unshaped_output_branch_p.size()[2])
        output_branch_d = unshaped_output_branch_d.view(-1, unshaped_output_branch_d.size()[2])
        
        # Also do this for inputs
        #input_branching = input_branching[:, :batch_size, :].detach()
        input_branching_z = input_branching[:, :, 0]
        input_branching_t = input_branching[:, :, 1]
        input_branching_p = input_branching[:, :, 2]
        input_branching_d = input_branching[:, :, 3]

        input_branching_z = torch.reshape(input_branching_z, (-1,))
        input_branching_t = torch.reshape(input_branching_t, (-1,))
        input_branching_p = torch.reshape(input_branching_p, (-1,))
        input_branching_d = torch.reshape(input_branching_d, (-1,))
        #input_branching_z = input_branching_z.view(-1, 1)
        #input_branching_t = input_branching_t.view(-1, 1)
        #input_branching_p = input_branching_p.view(-1, 1)
        #input_branching_d = input_branching_d.view(-1, 1)
    
        #input_mother_id = input_mother_id[:, :batch_size].detach()
        input_mother_id = torch.reshape(input_mother_id, (-1,))
        #input_mother_id = input_mother_id.view(-1,)

        #input_branch_mask = input_branch_mask[:, :batch_size].detach()
        #input_branch_mask =input_branch_mask.view(-1,)
        input_branch_mask = torch.reshape(input_branch_mask, (-1,))
        
        input_very_ending = torch.FloatTensor([1] * batch_size).detach()
        #input_ending      = torch.FloatTensor(np.zeros((input_branch_mask.size()[0], 1))).detach()
        input_ending      = torch.FloatTensor([0] * output_end.size()[0]).detach()
        """
        print("#############################################")
        print("\n Listing loss ready size \n")
        print("#############################################")
    
        print("Size of output_end: {}".format(          output_end.size()))
        print("Size of output_very_end: {}".format(     output_very_end.size()))
        print("Size of output_mother: {}".format(       output_mother.size()))
        print("Size of output_branch_z: {}".format(     output_branch_z.size()))
        print("Size of output_branch_t: {}".format(     output_branch_t.size()))
        print("Size of output_branch_p: {}".format(     output_branch_p.size()))
        print("Size of output_branch_d: {}".format(     output_branch_d.size()))
        print("Size of input_branching_z: {}".format(  input_branching_z.size()))
        print("Size of input_branching_t: {}".format(  input_branching_t.size()))
        print("Size of input_branching_p: {}".format(  input_branching_p.size()))
        print("Size of input_branching_d: {}".format(  input_branching_d.size()))
        #print("Size of input_n_branchings: {}".format(  input_n_branchings.size()))
        print("Size of input_branching: {}".format(     input_branching.size()))
        print("Size of input_mother_id: {}".format(     input_mother_id.size()))
        print("Size of input_branch_mask: {}".format(   input_branch_mask.size()))
        """
        # Everything should be of right shape, going into losses now
        #print("output_mother ", output_mother.size())
        #torch.set_printoptions(edgeitems=1000)
        #print("output_mother \n", output_mother)
        #print("input_branch_mask \n", input_branch_mask)
        
        #print("input_mother_id ", input_mother_id.size())
        unmasked_p_mother_output_loss   = self.loss_modules["p_mother"](output_mother, input_mother_id.long())
        unmasked_p_end_output_loss      = self.loss_modules["p_end"](output_end, input_ending)
        unmasked_p_branch_z_output_loss = self.loss_modules["p_branch_z"](output_branch_z, input_branching_z.long())
        unmasked_p_branch_t_output_loss = self.loss_modules["p_branch_t"](output_branch_t, input_branching_t.long())
        unmasked_p_branch_p_output_loss = self.loss_modules["p_branch_p"](output_branch_p, input_branching_p.long())
        unmasked_p_branch_d_output_loss = self.loss_modules["p_branch_d"](output_branch_d, input_branching_d.long())
        p_very_end_output_loss = self.loss_modules["p_end"](output_very_end, input_very_ending)
        
        # Mask the values that correspond to padded recurrence step
        p_mother_output_loss   = torch.masked_select(unmasked_p_mother_output_loss, input_branch_mask)
        p_end_output_loss      = torch.masked_select(unmasked_p_end_output_loss, input_branch_mask)
        p_branch_z_output_loss = torch.masked_select(unmasked_p_branch_z_output_loss, input_branch_mask)
        p_branch_t_output_loss = torch.masked_select(unmasked_p_branch_t_output_loss, input_branch_mask)
        p_branch_p_output_loss = torch.masked_select(unmasked_p_branch_p_output_loss, input_branch_mask)
        p_branch_d_output_loss = torch.masked_select(unmasked_p_branch_d_output_loss, input_branch_mask)
        #p_very_end_output_loss no need to mask, only one such item per batch sample.

        """
        print("#############################################")
        print("\n Listing loss output size \n")
        print("#############################################")

        print("Size of p_mother_output_loss: {}".format(    p_mother_output_loss.size()))
        print("Size of p_end_output_loss: {}".format(       p_end_output_loss.size()))
        print("Size of p_very_end_output_loss: {}".format(  p_very_end_output_loss.size()))
        print("Size of p_branch_z_output_loss: {}".format( p_branch_z_output_loss.size()))
        print("Size of p_branch_t_output_loss: {}".format( p_branch_t_output_loss.size()))
        print("Size of p_branch_p_output_loss: {}".format( p_branch_p_output_loss.size()))
        print("Size of p_branch_d_output_loss: {}".format( p_branch_d_output_loss.size()))
        """
        
        # Some special cases where the loss has to be produced following a certain fashion. First special case is if test_bool is true. Second is when binary_target is true
        
        if test_bool:
            # You want two return the batch loss and a list of tuples (one for each event) with the loss of the event and the loss per node. This is only really necessary when plotting a junipr tree and should be avoided.
            batch_loss_tensor = torch.empty((batch_size, 1))
            list_probabilities_per_jet = list()
            # Combine and sum (note that in this case p_very_end_output_loss has to be managed seperately because it is of a different size.
            list_losses = [p_mother_output_loss, p_end_output_loss, p_branch_z_output_loss,
                           p_branch_t_output_loss, p_branch_p_output_loss, p_branch_d_output_loss]
                
            losses_combined = torch.stack(list_losses, dim = 0)
            #print("losses_combined after stack: ", losses_combined.size())
            losses_combined = losses_combined.sum(dim=0)
            #print("losses_combined after sum: ", losses_combined.size())
            running_branching_count = 0
            
            for sample_id in range(batch_size):
                n_branching_in_sample = input_n_branchings[sample_id].item()
                loss_per_node = losses_combined[running_branching_count:(running_branching_count+n_branching_in_sample)] # these gives the loss at each node.
                loss_sample = float(loss_per_node.sum() + p_very_end_output_loss[sample_id]) # this is the loss of the sample
                batch_loss_tensor[sample_id] = loss_sample
                
                list_loss_per_nodes = loss_per_node.tolist()    # turn the loss per node into a list
                #print("loss_per_node: ", loss_per_node)
                #print("list_loss_per_nodes: ", list_loss_per_nodes)
                #print("Sample {} n_branching {} from initial {}".format(sample_id, n_branching_in_sample, running_branching_count))
                running_branching_count += n_branching_in_sample
                list_probabilities_per_jet.append( (loss_sample, list_loss_per_nodes) )
            
            if binary_target:
                # In this specific case, everything is similar, you just want the batch loss as a tensor for each sample (which you already did):
                return batch_loss_tensor, list_probabilities_per_jet
            batch_loss = batch_loss_tensor.sum().item()
            return batch_loss, list_probabilities_per_jet

        elif binary_target:
            # You do not want to output the loss as a float, you want the loss per event in a tensor batch_loss_tensor
            batch_loss_tensor = torch.empty((batch_size, 1))
            # Combine and sum (note that in this case p_very_end_output_loss has to be managed seperately because it is of a different size.
            list_losses = [p_mother_output_loss, p_end_output_loss, p_branch_z_output_loss,
                           p_branch_t_output_loss, p_branch_p_output_loss, p_branch_d_output_loss]
                
            losses_combined = torch.stack(list_losses, dim = 0)
            #print("losses_combined after stack: ", losses_combined.size())
            losses_combined = losses_combined.sum(dim=0)
            #print("losses_combined after sum: ", losses_combined.size())
            running_branching_count = 0
            for sample_id in range(batch_size):
                n_branching_in_sample = input_n_branchings[sample_id].item()
                batch_loss_tensor[sample_id] = losses_combined[running_branching_count:(running_branching_count+n_branching_in_sample)].sum() + p_very_end_output_loss[sample_id]
                #print("Sample {} n_branching {} from initial {}".format(sample_id, n_branching_in_sample, running_branching_count))
                running_branching_count += n_branching_in_sample
            return batch_loss_tensor
                
        else:
            # You just want the loss on the batch
            list_losses = [p_mother_output_loss, p_end_output_loss, p_very_end_output_loss, p_branch_z_output_loss,
                           p_branch_t_output_loss, p_branch_p_output_loss, p_branch_d_output_loss]
            losses_combined = torch.cat(list_losses, dim = 0).sum(dim=0)
            return losses_combined
    

    def compute_losses(self, batch_input, batch_output, test_bool = False, binary_target = False):
        """
        To compute the loss in a sequential way: DEPRECATED. Better to use compute_batch_losses
        
        batch_input, batch_output are dictionnaries with the entries required.
        
        If test_bool is false, returns the batch loss.
        If test_bool is true, returns a batch_loss and a list of tuples (one tuple for each batch item): (item probability, list of probability at each node).
        """
        #start_unary_loss = time.process_time()
        batch_loss = 0
        # Input info needed
        input_n_branchings = batch_input["n_branchings"]
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
        input_branching = input_branching[:, :output_mother.size()[1], :].detach()
        input_mother_id = input_mother_id[:, :output_mother.size()[1]].detach()
        
        #element_input_very_ending = torch.tensor(np.ones((1,), dtype=int), requires_grad=False).long()
        element_input_very_ending = torch.FloatTensor([1])
        element_input_very_ending.detach()
   
        if test_bool:
            list_probabilities_per_jet = list()
        
        if binary_target:
            batch_loss_tensor = torch.empty((input_n_branchings.size()[0], 1))
        
        #before_loop_unary_loss = time.process_time()
        #print("time to unary loss before loop {}".format(before_loop_unary_loss - start_unary_loss))
        for batch_element in range(input_n_branchings.size()[0]):
            #in_loop_unary_loss = time.process_time()
            
            element_n_branching       = input_n_branchings[batch_element].item()
            # consider the next item by restricting them to the real size of the sequence: element_n_branching.
            element_input_branching   = input_branching[batch_element, :element_n_branching, :]
            element_input_mother_id   = input_mother_id[batch_element, :element_n_branching]
            
            element_output_end        = output_end[batch_element, :element_n_branching, :]
            element_output_very_end   = output_very_end[batch_element, 0, :] #there is a second component of size 1 that is due to the RNN structure (fixed to 1 here)
            element_output_mother     = output_mother[batch_element, :element_n_branching, :element_n_branching]
            element_output_branch_z   = output_branch_z[batch_element, :element_n_branching, :]
            element_output_branch_t   = output_branch_t[batch_element, :element_n_branching, :]
            element_output_branch_p   = output_branch_p[batch_element, :element_n_branching, :]
            element_output_branch_d   = output_branch_d[batch_element, :element_n_branching, :]
            
            # tensor representing ending value (0 go on, 1 means end but the 1 is contained in element_input_very_ending defined above).
            #element_input_ending = torch.tensor(np.zeros((element_n_branching, 1)), requires_grad=False)
            element_input_ending = torch.FloatTensor(np.zeros((element_n_branching, 1)))
            element_input_ending.detach()
            
            #in_loop_unary_loss_first_trim = time.process_time()
            #print("Loss Loop: first trim {}".format(in_loop_unary_loss_first_trim - in_loop_unary_loss))
            p_mother_output_loss   = self.loss_modules["p_mother"](element_output_mother, element_input_mother_id.long())
            p_end_output_loss      = self.loss_modules["p_end"](element_output_end[:, 0], element_input_ending[:, 0])
            p_very_end_output_loss = self.loss_modules["p_end"](element_output_very_end, element_input_very_ending)
            p_branch_z_output_loss = self.loss_modules["p_branch_z"](element_output_branch_z, element_input_branching[:, 0].long())
            p_branch_t_output_loss = self.loss_modules["p_branch_t"](element_output_branch_t, element_input_branching[:, 1].long())
            p_branch_p_output_loss = self.loss_modules["p_branch_p"](element_output_branch_p, element_input_branching[:, 2].long())
            p_branch_d_output_loss = self.loss_modules["p_branch_d"](element_output_branch_d, element_input_branching[:, 3].long())
            
            #in_loop_unary_loss_loss_computed = time.process_time()
            #print("Loss Loop: loss computed {}".format(in_loop_unary_loss_loss_computed - in_loop_unary_loss_first_trim))
            
            list_losses = [p_mother_output_loss, p_end_output_loss, p_very_end_output_loss, p_branch_z_output_loss,
                           p_branch_t_output_loss, p_branch_p_output_loss, p_branch_d_output_loss]
            
            losses_combined = torch.cat(list_losses, dim = 0).sum(dim=0)
            #print("In loop loss, does losses_combined required grad ? {}".format(losses_combined.requires_grad))
            #in_loop_unary_loss_loss_compiled = time.process_time()
            #print("Loss Loop: loss compiled {}".format(in_loop_unary_loss_loss_compiled - in_loop_unary_loss_loss_computed))
            
            if test_bool:
                batch_element_loss_list = list()
                for node_counter in range(element_n_branching):
                    node_probability = p_mother_output_loss[node_counter] + p_end_output_loss[node_counter] + p_branch_z_output_loss[node_counter] + p_branch_t_output_loss[node_counter] + p_branch_p_output_loss[node_counter] + p_branch_d_output_loss[node_counter]
                    node_probability = node_probability.item()
                    batch_element_loss_list.append(node_probability)
                list_probabilities_per_jet.append( (float(losses_combined), batch_element_loss_list) )
                                                  
            batch_loss += losses_combined
            if binary_target:
                batch_loss_tensor[batch_element] = losses_combined
                    #print("The loss {} and what is saved {}".format(losses_combined, batch_loss_tensor[batch_element]))
        
            #print("In junipr runner loss module compute, global loss : {}".format(losses_combined))
            #in_loop_unary_loss_end = time.process_time()
            #print("Loss Loop: time to end of loop from last {}".format(in_loop_unary_loss_end - in_loop_unary_loss_loss_compiled))
        #end_loop_unary_loss = time.process_time()
        #print("time to end the loop in unary loss {}".format(end_loop_unary_loss - before_loop_unary_loss))
        if test_bool:
            if binary_target:
                 return batch_loss_tensor, list_probabilities_per_jet
            return float(batch_loss), list_probabilities_per_jet
        if binary_target:
            return batch_loss_tensor
        return batch_loss
    
    def compute_batch_binary_losses(self, batch_input, batch_output_quark, batch_output_gluon, val_loss = False, test_loss = False, detail_loss = False):
        """
        To compute the mean binary loss in the batch-leaning environment (note that this is the MEAN loss computed here and it should thus NOT be averaged over batch size)
        
        The argument batch_input, batch_output are dictionnaries with the entries required.
        
        If test_bool = True, returns the batch_loss and the sigmoid of the likelihood ratio + used and true labels in accuracy_info (demands a bathsize of 1 for the test set).
        
        If detail_loss = True, return the batch loss and a list of tuples (one tuple for each batch item): (item probability, list of probability at each node).
        
        If val_loss = True, returns an accuracy matching the one of test_bool plus a rough estimate of the real accuracy (if this one and test_bool are false, accuracy_info returns the accuracy by simple rounding)
        
        This implementation assumes equal likelihood of having a quark or a gluon jet (this should have been forced at dataloader level). It computes a sigmoid on log( Proba quark) - log (Proba gluon)
        
        Note that self.compute_batch_losses returns probabilities as -log(Proba).
        """
        start_loss = time.process_time()
        batch_loss = 0
        true_label = batch_input["label"].float().detach()  # the MC label telling you if it is a quark (1) gluon (0). Note that -1 values should have been handled by the dataset
        size_batch = true_label.size()[0]
        true_label = torch.reshape(true_label, (size_batch, ))
        # Compute the unary losses using the compute_batch_losses
        if detail_loss:
            tensor_proba_quark, list_proba_quark = self.compute_batch_losses(batch_input = batch_input, batch_output = batch_output_quark, test_bool = True, binary_target = True)
            tensor_proba_gluon, list_proba_gluon = self.compute_batch_losses(batch_input = batch_input, batch_output = batch_output_gluon, test_bool = True, binary_target = True)
        
        else:
            #start_q_loss = time.process_time()
            tensor_proba_quark = self.compute_batch_losses(batch_input = batch_input, batch_output = batch_output_quark, binary_target = True)
            #start_g_loss = time.process_time()
            tensor_proba_gluon = self.compute_batch_losses(batch_input = batch_input, batch_output = batch_output_gluon, binary_target = True)
            #finish_g_loss = time.process_time()
            #if not(val_loss):
            #print("Time to do quark loss {} and gluon loss {}".format(start_g_loss - start_q_loss, finish_g_loss - start_g_loss))
        
        # Compute the real proba ratio (in log).
        tensor_proba_ratio = torch.reshape(- tensor_proba_quark + tensor_proba_gluon, (size_batch, ))
        
        if self.binary_BCE_loss:
            # Use the real proba ratio in log form in the discriminator (BCE with the truth label and applying the mean).
            batch_loss = self.loss_modules["discriminator"](tensor_proba_ratio, true_label)
        else:
            # This maximises directly the likelihood ratio by minimising the opposite likelihood_ratio.
            sign_label = (-true_label) * 2 + 1 # maps the label 1 to -1 and 0 to 1
            batch_loss = torch.sigmoid(tensor_proba_ratio * sign_label).mean()  #correct the sign of the ratio depending on the label and applies a sigmoid as well as taking the mean.

        # Now gauge accuracy
        tensor_sigmoid_ratio = torch.sigmoid(tensor_proba_ratio).detach()
        if test_loss:
            # Your batch is of size 1. Here, you threshold the label in the test loop.
            accuracy_info = (tensor_sigmoid_ratio.item(), true_label.item(), batch_input["original_label"].item())

        else:
            tensor_predict_labels = torch.round(tensor_sigmoid_ratio)
            correct_count = (tensor_predict_labels == true_label).sum().item()
            if val_loss:
                accuracy_info = (correct_count / size_batch, tensor_predict_labels, true_label)
            else:
                accuracy_info = correct_count / size_batch

        if detail_loss:     # Case where you analyse the probability per node.
            detail_list_proba = list()
            for sample_id in range(size_batch):
                _, sample_proba_per_nodes_q = list_proba_quark[sample_id]
                _, sample_proba_per_nodes_g = list_proba_gluon[sample_id]
                ratio_proba = list()
                for node_id in range(len(sample_proba_per_nodes_q)):
                    ratio_proba.append(-sample_proba_per_nodes_q[node_id] + sample_proba_per_nodes_g[node_id])
                detail_list_proba.append((tensor_proba_ratio[sample_id], ratio_proba))
            return batch_loss, detail_list_proba, accuracy_info
        else:
            #print("Time to do the rest {}".format(time.process_time() - start_loss))
            return batch_loss, accuracy_info

    def compute_binary_losses(self, batch_input, batch_output_quark, batch_output_gluon, val_loss = False, test_loss = False, detail_loss = False):
        """
        To compute the binary loss in our batch-leaning environment: the sequences are flatten into an array.
          
        batch_input, batch_output are dictionnaries with the entries required.
          
        Note that the ignore_index specification from the nn.CrossEntropyLoss() is not sufficient here: padded values are passed through the networks.
          
        If test_loss is false, returns the batch loss.
        If test_loss is true, returns a batch_loss and a list of tuples (one tuple for each batch item): (item probability, list of probability at each node).
        
        This implementation assumes equal likelihood of having a quark or a gluon jet. It computes a sigmoid on log( Proba quark) - log (Proba gluon)
        
        Note that self.compute_losses returns probabilities as -log(Proba).
        """
        #start_loss = time.process_time()
        batch_loss = 0
        #dataset_label = batch_input["dataset_label"] # a specific label that tells you whether the jet is from a quark (1) or gluon (0) rich table.
        true_label = batch_input["label"]   # the MC label telling you if it is a quark (1) gluon (0) or other (-1)
        size_batch = true_label.size()[0]
        
        if detail_loss:
            tensor_proba_quark, list_proba_quark = self.compute_losses(batch_input = batch_input, batch_output = batch_output_quark, test_bool = True, binary_target = True)
            tensor_proba_gluon, list_proba_gluon = self.compute_losses(batch_input = batch_input, batch_output = batch_output_gluon, test_bool = True, binary_target = True)
        
        else:
            #start_q_loss = time.process_time()
            #print("time to quark loss {}".format(start_q_loss - start_loss))
            tensor_proba_quark = self.compute_losses(batch_input = batch_input, batch_output = batch_output_quark, binary_target = True)
            #start_g_loss = time.process_time()
            #print("time to do quark loss {}".format(start_g_loss - start_q_loss))
            tensor_proba_gluon = self.compute_losses(batch_input = batch_input, batch_output = batch_output_gluon, binary_target = True)
            #finish_g_loss = time.process_time()
            #print("time to do gluon loss {}".format(finish_g_loss - start_g_loss))
        
        if test_loss:
            prediction_label = 0
            t_label = 0
            u_label = 0
        elif val_loss:
            prediction_label = torch.tensor(np.zeros((size_batch, 1)), requires_grad=False)
            u_labels = torch.tensor(np.zeros((size_batch, 1)), requires_grad=False)
        
        detail_list_proba = list()
        
        correct_count = 0   # to accumulate number of correctly predicted labels.
        start_loop_loss = time.process_time()
        for indice in range(size_batch):
            #dataset_jet = dataset_label[indice].item()
            true_ID_jet = true_label[indice].item()
            used_label = true_ID_jet
            
            #if true_ID_jet == -1:
                # how to deal with mis classified labels from the testing set ?
                # In this scenario, if such jets are present, consider the label to be the dataset enriched one (quark or gluon).
                #used_label = dataset_jet
            used_label = torch.FloatTensor([used_label]).detach()
            
            if detail_loss:
                tuple_proba_quark = list_proba_quark[indice]
                tuple_proba_gluon = list_proba_gluon[indice]

                _, proba_quark_per_nodes = tuple_proba_quark
                _, proba_gluon_per_nodes = tuple_proba_gluon
            
            if test_loss:
                # Then you have a batchsize of 1
                proba_quark = torch.tensor([tensor_proba_quark.item()])
                proba_gluon = torch.tensor([tensor_proba_gluon.item()])
            else:
                proba_quark = tensor_proba_quark[indice]
                proba_gluon = tensor_proba_gluon[indice]
            
            # The following is for supervised learning.
            #prediction_loss = self.loss_modules["discriminator"](-proba_quark + proba_gluon, used_label.long())
            
            # The following for semi-supervised learning: maximise the likelihood ratio itself, with the sign depending on the label
            # We use a sigmoid to make the loss smooth.
            if self.binary_BCE_loss:
                proba_ratio = - proba_quark + proba_gluon # we need to compute the real proba ratio and use this in the discriminator (BCE with the truth label).
                proba = self.loss_modules["discriminator"](proba_ratio, used_label)
            
            else:
                print("WARNING: using the non appropriate loss.")
                # This is a dubious way of managing the loss that works though not great: it directly maximises an orientated version of the likelihood ratio (with the model associated with the truth info of the jet).
                if used_label == 1:
                    proba = torch.sigmoid(proba_quark - proba_gluon) #the real probability: take a minus in front of the proba of self.compute_losses. However, we want to maximise the result of the sigmoid, meaning  - the sigmoid.
                    #proba = proba_quark - proba_gluon
                else:
                    proba = torch.sigmoid(-proba_quark + proba_gluon)
                    #proba = -proba_quark + proba_gluon
            #print("proba_quark {}, proba_gluon {} and proba {} for used_label {}".format(proba_quark, proba_gluon, proba, used_label))
            batch_loss += proba

            # Predict a label based on the ratio test. Need to put - as proba_quark and proba_gluon are the -log(proba)
            if test_loss:
                prediction_label = torch.sigmoid(-proba_quark + proba_gluon).item()
                t_label = true_ID_jet
                u_label = used_label
            else:
                prediction = torch.sigmoid(-proba_quark + proba_gluon).detach()
                predicted_label = torch.round(prediction)
                
                if val_loss:
                    prediction_label[indice] = prediction
                    u_labels[indice] = used_label
                
                if predicted_label == used_label:
                    correct_count += 1

            if detail_loss:
                ratio_proba = list()
                for node in range(len(proba_quark_per_nodes)):
                    ratio_proba.append(-proba_quark_per_nodes[node] + proba_gluon_per_nodes[node])
                detail_list_proba.append((proba_ratio, ratio_proba))
        #end_loop_loss = time.process_time()
        #print("time to do the loop loss {}".format(end_loop_loss - start_loop_loss))
        if test_loss:
            # for test, the batch size is 1
            accuracy = (prediction_label, u_label, batch_input["original_label"].item())
        else:
            # rough idea of what the accuracy is and also a tensor with the prediction of earch label and each labels used_labels
            if val_loss:
                accuracy = (correct_count / size_batch, prediction_label, u_labels)
            else: # regular training
                accuracy = correct_count / size_batch
        return batch_loss, detail_list_proba, accuracy
                  
                  
    def train(self):
        """
        The training loop. Note that manipulation of the input/output into the losses is very particular here.
        """
        # Set model to train mode
        self.JUNIPR_model.train()
        
        step_count = 0
        number_samples_processed = self.previously_trained_number_steps
        number_samples_processed_since_last = 0
        
        # This part sets up some logging information for later plots.
        if self.logger_data_bool:
            # initiate the logger
            write_to_file(os.path.join(self.result_path, "logger_info.txt"), ["#step", "train_loss", "test_loss"], action = 'w')
            write_to_file(os.path.join(self.result_path, "saved_info.txt"), ["#epoch", "step"], action = 'w')
            write_to_file(os.path.join(self.result_path, "logger_epoch_info.txt"), ["#epoch", "train_loss"], action = 'w')
                
        time_start_sequence = time.process_time()
        #torch.backends.cudnn.benchmark = True
        for epoch in range(1 + self.previously_trained_number_epoch, self.num_epochs + 1 + self.previously_trained_number_epoch):
            epoch_loss = 0
            batch_count = 0
            
            # If a batchsize schedule is used:
            if self.batchsize_scheduler_bool:
                if str(epoch) in self.batchsize_schedule:
                    self.reset_dataloader(self.batchsize_schedule[str(epoch)], epoch)
            
            # If a learning rate schedule is used:
            if self.lr_scheduler_bool:
                if str(epoch) in self.lr_schedule:
                    self.lr = self.lr_schedule[str(epoch)]
                    print("\n#######################################################\n")
                    print("Epoch {} | New learning rate : {}".format(epoch, self.lr))
                    print("\n#######################################################\n")
                    # Update the learning rate parameter of the optimiser.
                    for param_group in self.optimiser.param_groups:
                        param_group['lr'] = self.lr
            
            #time_start_load = time.process_time()
            #time_finish_load = 0
            for batch_input in self.train_dataloader:
                #time_finish_load = time.process_time()
                #print("time to load batches {}".format(time_finish_load - time_start_load))
                step_count += 1
                
                #very_start = time.process_time()
                #print("Starting step {}".format(step_count))
                input_n_branchings = batch_input["n_branchings"]
                
                size_batch = input_n_branchings.size()[0]
                batch_count += size_batch
                number_samples_processed_since_last += size_batch
                
                batch_output = self.JUNIPR_model(batch_input)
                #to_loss_time = time.process_time()
                #print("Time to reach loss step {}".format(to_loss_time - very_start))
                #batch_loss = self.compute_losses(batch_input, batch_output) / size_batch
                #time_first_loss = time.process_time()
                
                batch_loss = self.compute_batch_losses(batch_input, batch_output) / size_batch
                #batch_loss = self.compute_batch_losses(batch_input, batch_output, binary_target = True).sum()  / size_batch
                #batch_loss, tuple_per_node = self.compute_batch_losses(batch_input, batch_output, test_bool = True)
                #batch_loss, tuple_per_node = self.compute_batch_losses(batch_input, batch_output, test_bool = True, binary_target = True)
                #time_second_loss = time.process_time()
               
                #batch_loss = self.compute_losses(batch_input, batch_output) / size_batch
                
                #print("Time to compute first loss {} and second loss {}".format(time_second_loss - time_first_loss, loss_time - time_second_loss ))
                # print("Batch loss with batch oriented {} and sequential {}".format(batch_loss1, batch_loss2))
                #print("Batch loss with batch {}".format(batch_loss))
                
                #loss_time = time.process_time()
                #print("Time to compute loss step {} with value {}".format(loss_time - time_first_loss, batch_loss))
                
                epoch_loss += float(batch_loss * size_batch)
                
                self.optimiser.zero_grad()
                batch_loss.backward()
                self.optimiser.step()
                #print("Time to load batch step {}".format(time_finish_load - time_start_load))
                #print("Time to reach loss step {}".format(to_loss_time - time_finish_load))
                #print("Time to compute loss step {}".format(loss_time - to_loss_time))
                #print("Time to backprop loss step {}".format(time.process_time() - loss_time))
                #print("Finished step {} |Time for whole step {}".format(step_count, time.process_time() - very_start))
                if (int(number_samples_processed_since_last) >= int(self.test_frequency * 200)):
                    print("Training {} step | batch loss : {}".format(step_count, float(batch_loss)))
                    # Report result to TensorBoard
                    self.writer.add_scalar("training_loss", float(batch_loss), step_count)
                    #time_start_test_loop = time.process_time()
                    test_loss = self.test_loop(step=step_count)
                    time_end_sequence = time.process_time()
                    #print("Time it took for the test lopp: {}".format(time_end_sequence - time_start_test_loop))
                    print("Time it took for the sequence: {}".format(time_end_sequence - time_start_sequence))
                    number_samples_processed += number_samples_processed_since_last
                    number_samples_processed_since_last = 0
                    
                    if self.logger_data_bool:
                        # append to the logger
                        write_to_file(os.path.join(self.result_path, "logger_info.txt"), [int(number_samples_processed / (self.test_frequency * 200)), float(batch_loss), float(test_loss)], action = 'a', limit_decimal = True)
                    self.JUNIPR_model.train()
                    time_start_sequence = time.process_time()
                #time_start_load = time.process_time()
            if (epoch % 1 == 0 and self.save_model_bool):
                self.JUNIPR_model.save_model(self.result_path)
                if self.logger_data_bool:
                    # append to the logger to confirm the last updates
                    write_to_file(os.path.join(self.result_path, "saved_info.txt"), [epoch, step_count], action = 'a')
            if self.logger_data_bool:
                write_to_file(os.path.join(self.result_path, "logger_epoch_info.txt"), [epoch, epoch_loss/batch_count], action = 'a')
            print("\n#######################################################\n")
            print("Epoch {} | loss : {}".format(epoch, epoch_loss/batch_count))
            print("\n#######################################################\n")
                        
    def binary_train(self):
        """
        A binary trainer. Requires one model for quarks and another one for gluons.
        Uses the binary objective.
        """
        # Set model to train mode
        self.JUNIPR_quark_model.train()
        self.JUNIPR_gluon_model.train()
        
        step_count = 0
        number_samples_processed = 0
        number_samples_processed_since_last = 0 #warning, to load the first binary you need some adapation
        
        # This part sets up some logging information for later plots.
        if self.logger_data_bool:
            # initiate the logger
            write_to_file(os.path.join(self.result_path, "logger_info.txt"), ["#step", "train_loss", "test_loss", "train_acc", "test_acc", "test_auc"], action = 'w')
            write_to_file(os.path.join(self.result_path, "saved_info.txt"), ["#epoch", "step"], action = 'w')
            write_to_file(os.path.join(self.result_path, "logger_epoch_info.txt"), ["#epoch", "train_loss", "train_acc"], action = 'w')
        
        torch.backends.cudnn.benchmark = True
        time_start_sequence = time.process_time()
        for epoch in range(1 + self.previously_trained_number_epoch, self.num_epochs + 1 + self.previously_trained_number_epoch):
            epoch_loss = 0
            epoch_acc = 0
            batch_count = 0
        
            # If a batchsize schedule is used:
            if self.batchsize_scheduler_bool:
                if str(epoch) in self.batchsize_schedule:
                    self.reset_dataloader(self.batchsize_schedule[str(epoch)], epoch)
            
            # If a learning rate schedule is used:
            if self.lr_scheduler_bool:
                if str(epoch) in self.lr_schedule:
                    self.lr = self.lr_schedule[str(epoch)]
                    print("\n#######################################################\n")
                    print("Epoch {} | New learning rate : {}".format(epoch, self.lr))
                    print("\n#######################################################\n")
                    # Update the learning rate parameter of the optimiser.
                    for param_group in self.optimiser.param_groups:
                        param_group['lr'] = self.lr
        
            time_start_load = time.process_time()
            for batch_input in self.train_dataloader:
                time_finish_load = time.process_time()
                #print("time to load batches {}".format(time_finish_load - time_start_load))
                #very_start = time.process_time()
                #print("Starting step {}".format(step_count))
                step_count += 1
                input_n_branchings = batch_input["n_branchings"]
                size_batch = input_n_branchings.size()[0]
                batch_count += size_batch
                number_samples_processed_since_last += size_batch
                
                batch_output_quark = self.JUNIPR_quark_model(batch_input)
                #print("Going in gluon")
                batch_output_gluon = self.JUNIPR_gluon_model(batch_input)
                #print("Going in loss")
                to_loss_time = time.process_time()
                #print("Time to reach loss step {}".format(to_loss_time - very_start))
                
                #batch_loss, accuracy_batch = self.compute_batch_binary_losses(batch_input, batch_output_quark, batch_output_gluon)
                batch_loss, _, accuracy_batch = self.compute_binary_losses(batch_input, batch_output_quark, batch_output_gluon)
                """
                batch_loss2, _, accuracy_batch2 = self.compute_binary_losses(batch_input, batch_output_quark, batch_output_gluon)
                print("Batch loss with batch oriented {} and sequential {}".format(batch_loss, batch_loss2))
                print("Batch accuracy with batch oriented {} and sequential {}".format(accuracy_batch, accuracy_batch2))
                """
                loss_time = time.process_time()
                
                #batch_loss = batch_loss / size_batch   # batch binary is already averaging the loss.
                self.optimiser.zero_grad()
                batch_loss.backward()
                self.optimiser.step()
                time_finish_backprop = time.process_time()
                epoch_loss += float(batch_loss * size_batch)
                epoch_acc += float(accuracy_batch * size_batch)
                print("Time to load batch step {}".format(time_finish_load - time_start_load))
                print("Time to reach loss step {}".format(to_loss_time - time_finish_load))
                print("Time to compute loss step {}".format(loss_time - to_loss_time))
                print("Time to finish backprop loss step {}".format(time_finish_backprop - loss_time))
                #print("Finished step {} |Time for whole step {}".format(step_count, time.process_time() - very_start))
                
                if (int(number_samples_processed_since_last) >= int(self.test_frequency * 200)):
                    print("Training {} step | batch loss : {} | accuracy : {}".format(step_count, float(batch_loss), float(accuracy_batch)))
                    # Report result to TensorBoard
                    self.writer.add_scalar("training_loss", float(batch_loss/ size_batch), step_count)
                    #time_start_test_loop = time.process_time()
                    test_loss, test_acc, test_auc = self.binary_test_loop(step=step_count, use_small_val = True)
                    time_end_sequence = time.process_time()
                    #print("Time it took for the test loop: {}".format(time_end_sequence - time_start_test_loop))
                    print("Time it took for the sequence: {}".format(time_end_sequence - time_start_sequence))
                    
                    number_samples_processed += number_samples_processed_since_last
                    number_samples_processed_since_last = 0
                    if self.logger_data_bool:
                        # append to the logger
                        # for the first binary model training, need to replace step by just int(number_samples_processed)
                        write_to_file(os.path.join(self.result_path, "logger_info.txt"), [int(number_samples_processed / (self.test_frequency * 200)), float(batch_loss), float(test_loss), float(accuracy_batch), float(test_acc), float(test_auc)], action = 'a', limit_decimal = True)
                    self.JUNIPR_quark_model.train()
                    self.JUNIPR_gluon_model.train()
                    time_start_sequence = time.process_time()
                time_start_load = time.process_time()
            if (epoch % 1 == 0 and self.save_model_bool):
                self.JUNIPR_quark_model.save_model(os.path.join(self.result_path, "quark_model"))
                self.JUNIPR_gluon_model.save_model(os.path.join(self.result_path, "gluon_model"))
                if self.logger_data_bool:
                    # append to the logger to confirm the last updates
                    write_to_file(os.path.join(self.result_path, "saved_info.txt"), [epoch, step_count], action = 'a')
            if self.logger_data_bool:
                write_to_file(os.path.join(self.result_path, "logger_epoch_info.txt"), [epoch, epoch_loss/batch_count, epoch_acc/batch_count], action = 'a')
            print("\n#######################################################\n")
            print("Epoch {} | loss : {} | accuracy : {}".format(epoch, epoch_loss/batch_count, epoch_acc/batch_count))
            print("\n#######################################################\n")
                  
    def binary_test_loop(self, step:int, use_small_val = False):
        self.JUNIPR_quark_model.eval()
        self.JUNIPR_gluon_model.eval()
        """
        get_an_event = True
        size_sample = len(self.test_dataloader)
        index = random.randint(0, size_sample - 1)
        """
        with torch.no_grad():
            #time_very_start_test_loop = time.process_time()
            output_loss = 0
            output_accuracy = 0
            total_size = 0
            accumulate_tensors_pred = list()
            accumulate_tensors_label = list()
            if use_small_val:
                dataloader_to_use = self.small_validation_dataloader
            else:
                dataloader_to_use = self.validation_dataloader
            #time_start_test_loop = time.process_time()
            #time_loading_batch = time_start_test_loop
            for batch_count, batch_input in enumerate(dataloader_to_use):
                #begin_loop_time = time.process_time()
                #batch_input = batch_input.to(self.device)
                input_n_branchings = batch_input["n_branchings"]
                size_batch = input_n_branchings.size()[0]
                total_size += size_batch
                
                batch_output_quark = self.JUNIPR_quark_model(batch_input)
                batch_output_gluon = self.JUNIPR_gluon_model(batch_input)
                #to_loss_time = time.process_time()
                #batch_loss, _, accuracy = self.compute_binary_losses(batch_input, batch_output_quark, batch_output_gluon, val_loss = True)
                batch_loss, accuracy = self.compute_batch_binary_losses(batch_input, batch_output_quark, batch_output_gluon, val_loss = True)
                #loss_time = time.process_time()
                
                """
                batch_loss2, _, accuracy_batch2 = self.compute_binary_losses(batch_input, batch_output_quark, batch_output_gluon, val_loss = True)
                print("Batch loss with batch oriented {} and sequential {}".format(batch_loss, batch_loss2/size_batch))
                print("Batch accuracy with batch oriented {} and sequential {}".format(accuracy_batch, accuracy_batch2))
                """
                
                output_loss += (batch_loss * size_batch)
                accuracy, prediction_label, u_labels = accuracy
                accumulate_tensors_pred.append(prediction_label)
                accumulate_tensors_label.append(u_labels)
                
                output_accuracy += accuracy * size_batch
                #formating_time = time.process_time()
                #print("Time laoding batch {} then for reaching loss {} then time loss in Test {} and for formatting it {}".format(begin_loop_time - time_loading_batch, to_loss_time - begin_loop_time, loss_time - to_loss_time, formating_time - loss_time))
                #time_loading_batch = time.process_time()
                """
                if batch_count == index:
                    _, event_probability_list = self.compute_batch_binary_losses(batch_input, batch_output_quark, batch_output_gluon, test_loss = True)
                    self.log_jet_constructed(batch_input, event_probability_list, step = step)
                    #get_an_event = False
                """
            mean_loss = output_loss / total_size
            mean_accuracy = output_accuracy / total_size
            #time_end_essential = time.process_time()
            
            full_pred = torch.flatten(torch.cat(accumulate_tensors_pred, dim = 0)).numpy()
            full_label = torch.flatten(torch.cat(accumulate_tensors_label, dim = 0)).numpy().astype(int)
            auc_validation = metrics.roc_auc_score(full_label, full_pred)
            """
            time_end_test_loop = time.process_time()
            print("Time in TEST LOOP the initial part {}".format(time_start_test_loop - time_very_start_test_loop))
            print("Time in TEST LOOP to do the essential {}".format(time_end_essential - time_start_test_loop))
            print("Time in TEST LOOP to do the auc {}".format(time_end_test_loop - time_end_essential))
            """
            print("Validation {} step | batch loss = {} | accuracy = {} | AUC = {}".format(step, float(mean_loss), float(mean_accuracy), auc_validation))
            self.writer.add_scalar("validation_loss", float(mean_loss), step)
        return mean_loss, mean_accuracy, auc_validation

    def test_loop(self, step:int):
        """
        get_an_event = True
        size_sample = len(self.test_dataloader)
        index = random.randint(0, size_sample - 1)
        """
        self.JUNIPR_model.eval()
        with torch.no_grad():
            output_loss = 0
            total_size = 0
            for batch_count, batch_input in enumerate(self.validation_dataloader):
                  #batch_input = batch_input.to(self.device)
                  input_n_branchings = batch_input["n_branchings"]
                  size_batch = input_n_branchings.size()[0]
                  total_size += size_batch
                  
                  batch_output = self.JUNIPR_model(batch_input)
          
                  batch_loss = self.compute_batch_losses(batch_input, batch_output)
                  output_loss += float(batch_loss)
                  """
                  if batch_count == index:
                    _, event_probability_list = self.compute_losses(batch_input, batch_output, test_bool = True)
                    self.log_jet_constructed(batch_input, event_probability_list, step = step)
                    #get_an_event = False
                 """
            mean_loss = output_loss / total_size
                  
            print("Validation {} step | loss = {}".format(step, float(mean_loss)))
            self.writer.add_scalar("validation_loss", float(mean_loss), step)
        return mean_loss

    def log_jet_constructed(self, event_input, event_probability_list, step=0, path = None, tensorboard_logger_bool = True, binary_junipr_bool = False):
        """
        Receiving a specific event information, calls tree_plotter from utils to draw the tree and log it to tensorboard.
        I advise against using this in training as it significantly slows operation
        """
        information_dictionary = dict()
        information_dictionary["label"]    = event_input["label"]
        information_dictionary["branching"]    = event_input["unscaled_branching"][0, :, :]
        information_dictionary["n_branchings"] = event_input["n_branchings"]
        information_dictionary["CSJets"]       = event_input["CSJets"][0, :, :]
        information_dictionary["probability_list"] = event_probability_list
        information_dictionary["CS_ID_mothers"]    = event_input["CS_ID_mothers"][0, :]
        information_dictionary["CS_ID_daugthers"]  = event_input["CS_ID_daugthers"][0, :, :]
        information_dictionary["daughter_momenta"] = event_input["daughter_momenta"][0, :, :]
                  
        if tensorboard_logger_bool:
            figure = tree_plotter(information_dictionary, path = "", segment_size = 2.0, return_plt_bool = True)
            # Save the figure as a PNG in memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(figure)   #figure needs to be closed to avoid spurious display
            buf.seek(0)
            # convert PNG buffer to TF image
            #image = tf.image.decode_png(buf.getvalue(), channels=4)
            self.writer.add_figure("Junipr Tree", figure, step)
        else:
            figure = tree_plotter(information_dictionary, path = path, segment_size = 2.0, return_plt_bool = False)

    def assess_unary_JUNIPR_proba_distributions(self):
        """
            This code iterates through the dataset to form the following histograms:
            - ending size (number of nodes)
            - mother id in energy ordered list
            - branch_z
            - branch_theta
            - branch_phi
            - branch_delta
        It performs this for both the JUNIPR model predictions and the real data so that the quality of the match can be assessed.
        The results are stored as histograms.
        """
        save_path = os.path.join(self.result_path, 'probability_distributions')
        os.makedirs(save_path, exist_ok=True)
        print(save_path)
        self.JUNIPR_model.eval()
        with torch.no_grad():
            #apply_softmax_dim1 = nn.Softmax(dim=1)
            apply_softmax_dim2 = nn.Softmax(dim=2)
            apply_sigmoid = nn.Sigmoid()
            
            input_ending_count = np.zeros(self.padding_size).astype('int')
            input_mother_count = np.zeros(self.padding_size).astype('int')
            input_branch_z_count = np.zeros(self.granularity).astype('int')
            input_branch_t_count = np.zeros(self.granularity).astype('int')
            input_branch_p_count = np.zeros(self.granularity).astype('int')
            input_branch_d_count = np.zeros(self.granularity).astype('int')
            
            output_ending_count = np.zeros(self.padding_size).astype('float')
            output_mother_count = np.zeros(self.padding_size).astype('float')
            output_branch_z_count = np.zeros(self.granularity).astype('float')
            output_branch_t_count = np.zeros(self.granularity).astype('float')
            output_branch_p_count = np.zeros(self.granularity).astype('float')
            output_branch_d_count = np.zeros(self.granularity).astype('float')
            
            for batch_count, batch_input in enumerate(self.test_dataloader):
                model_batch_output = self.JUNIPR_model(batch_input)
                #batch_input, model_batch_output = self.dataloader.undo_mapping_FeatureScalingOwn(batch_input, model_batch_output)    # need to restore the output and the data to it's real format, not the processed one

                output_end      = model_batch_output["output_end"]
                output_very_end = model_batch_output["output_very_end"]
                output_mother   = model_batch_output["output_mother"]
                output_branch_z = model_batch_output["output_branch_z"]
                output_branch_t = model_batch_output["output_branch_t"]
                output_branch_p = model_batch_output["output_branch_p"]
                output_branch_d = model_batch_output["output_branch_d"]
                
                recurrent_limit = output_mother.size()[1]
                
                output_end = torch.cat([output_end, output_very_end], dim = 1)[:, :, 0]
                
                output_end      = apply_sigmoid(output_end)[0, :].numpy()
                output_mother   = apply_softmax_dim2(output_mother)[0, :, :].numpy()
                output_branch_z = apply_softmax_dim2(output_branch_z)[0, :, :].numpy()
                output_branch_t = apply_softmax_dim2(output_branch_t)[0, :, :].numpy()
                output_branch_p = apply_softmax_dim2(output_branch_p)[0, :, :].numpy()
                output_branch_d = apply_softmax_dim2(output_branch_d)[0, :, :].numpy()
                
                input_end          = (batch_input["n_branchings"] + 1).numpy()     # +1 seems you end after having done all branching
                input_branching    = batch_input["branching"][0, :recurrent_limit, :]
                input_mother_id    = batch_input["mother_id_energy"][0, :recurrent_limit].numpy().astype('int')
                input_branch_mask  = batch_input["branching_mask"][0, :recurrent_limit].numpy().astype('int')
                input_branch_z     = input_branching[ :, 0].numpy().astype('int')
                input_branch_t     = input_branching[ :, 1].numpy().astype('int')
                input_branch_p     = input_branching[ :, 2].numpy().astype('int')
                input_branch_d     = input_branching[ :, 3].numpy().astype('int')

                np.add.at(input_ending_count, input_end, 1)
                np.add.at(input_mother_count, input_mother_id, 1)
                np.add.at(input_branch_z_count, input_branch_z, 1)
                np.add.at(input_branch_t_count, input_branch_t, 1)
                np.add.at(input_branch_p_count, input_branch_p, 1)
                np.add.at(input_branch_d_count, input_branch_d, 1)

                output_ending_count[1:(recurrent_limit+2)] += output_end[:] # there is a one since junipr starts predicting at node 1 at least.

                for i in range(recurrent_limit):
                    output_mother_count[:recurrent_limit] += output_mother[i, :]
                    output_branch_z_count[:] += output_branch_z[i, :]
                    output_branch_t_count[:] += output_branch_t[i, :]
                    output_branch_p_count[:] += output_branch_p[i, :]
                    output_branch_d_count[:] += output_branch_d[i, :]
                        
        input_ending_count = input_ending_count / input_ending_count.sum()
        input_mother_count = input_mother_count / input_mother_count.sum()
        input_branch_z_count =  input_branch_z_count /  input_branch_z_count.sum()
        input_branch_t_count =  input_branch_t_count /  input_branch_t_count.sum()
        input_branch_p_count =  input_branch_p_count /  input_branch_p_count.sum()
        input_branch_d_count =  input_branch_d_count /  input_branch_d_count.sum()
        
        output_ending_count = output_ending_count / output_ending_count.sum()
        output_mother_count = output_mother_count / output_mother_count.sum()
        output_branch_z_count = output_branch_z_count /  output_branch_z_count.sum()
        output_branch_t_count = output_branch_t_count /  output_branch_t_count.sum()
        output_branch_p_count = output_branch_p_count /  output_branch_p_count.sum()
        output_branch_d_count = output_branch_d_count /  output_branch_d_count.sum()
        

        plot_dictionary = {'ending':    (input_ending_count, output_ending_count),
                           'mother_id': (input_mother_count, output_mother_count),
                           'branch_z':  (input_branch_z_count, output_branch_z_count),
                           'branch_theta': (input_branch_t_count, output_branch_t_count),
                           'branch_phi'  : (input_branch_p_count, output_branch_p_count),
                           'branch_delta': (input_branch_d_count, output_branch_d_count)}
        
        for i, plot_name in enumerate(['ending', 'mother_id', 'branch_z', 'branch_theta', 'branch_phi', 'branch_delta']):
            print(plot_name)
            fig = plt.figure()
            plt.bar(*JUNIPR_distributions_plot.prepare_date(plot_dictionary[plot_name][0], plot_name), label ='Monte Carlo', **JUNIPR_distributions_plot.plot_end_setting[plot_name])
            plt.bar(*JUNIPR_distributions_plot.prepare_date(plot_dictionary[plot_name][1], plot_name), label ='JUNIPR', **JUNIPR_distributions_plot.plot_end_setting[plot_name])
            
            if JUNIPR_distributions_plot.plot_xscale[plot_name] == 'log':
                plt.xscale('log')
            
            if plot_name == 'mother_id':
                plt.xticks(np.arange(0, 15, 1))
            elif plot_name == 'ending':
                plt.xticks(JUNIPR_distributions_plot.plot_xticks[plot_name])
            else:
                plt.xticks(JUNIPR_distributions_plot.plot_xticks[plot_name][0], JUNIPR_distributions_plot.plot_xticks[plot_name][1])
            
            if JUNIPR_distributions_plot.plot_axis[plot_name]:
                plt.axis(JUNIPR_distributions_plot.plot_axis[plot_name])

            
            plt.xlabel(JUNIPR_distributions_plot.plot_xlabels[plot_name])
            plt.ylabel(JUNIPR_distributions_plot.plot_ylabels[plot_name])
            plt.legend()
            fig.savefig(os.path.join(save_path, plot_name + '_distribution.png'), dpi=300, format='png', bbox_inches='tight')
            plt.close()

    def run(self, config, train_bool = False, load_bool = False, assess_bool = False, print_jets_bool = False, save_model_bool = False, binary_junipr_bool = False):
        """
        The centre of the Junipr Runner. Runs the operations required by the configuration
        """
        if train_bool and not(binary_junipr_bool):
            # If need be, you have to find the last configuration in the batchsize and learning rates schedules and use these ones.

            if self.lr_scheduler_bool:
                keys_lr_schedule = np.array(list(self.lr_schedule.keys()), int)
                if len(keys_lr_schedule[keys_lr_schedule < self.previously_trained_number_epoch]) != 0:
                    last_epoch_config = keys_lr_schedule[keys_lr_schedule < self.previously_trained_number_epoch].max()
                    self.lr = self.lr_schedule[str(last_epoch_config)]
        
            if self.batchsize_scheduler_bool:
                keys_batchsize_schedule = np.array(list(self.batchsize_schedule.keys()), int)
                if len(keys_batchsize_schedule[keys_batchsize_schedule < self.previously_trained_number_epoch]) != 0:
                    last_epoch_config = keys_batchsize_schedule[keys_batchsize_schedule < self.previously_trained_number_epoch].max()
                    self.reset_dataloader(self.batchsize_schedule[str(last_epoch_config)], last_epoch_config)
            
            # training a single version of junipr
            self.train()
            if save_model_bool:
                  self.JUNIPR_model.save_model(self.result_path)
                  
        elif train_bool and binary_junipr_bool:
            # if reloading training (self.previously_trained_number_epoch > 0), get the last configuration from scheduler (lr and batchsize)
            if self.previously_trained_number_epoch != 0:
                print("The binary Junipr model has already been trained for {} epochs".format(self.previously_trained_number_epoch))
            if self.lr_scheduler_bool:
                keys_lr_schedule = np.array(list(self.lr_schedule.keys()), int)
                if len(keys_lr_schedule[keys_lr_schedule < self.previously_trained_number_epoch]) != 0:
                    last_epoch_config = keys_lr_schedule[keys_lr_schedule < self.previously_trained_number_epoch].max()
                    self.lr = self.lr_schedule[str(last_epoch_config)]
                print("\n#######################################################\n")
                print("Epoch {} | New learning rate : {}".format(self.previously_trained_number_epoch, self.lr))
                print("\n#######################################################\n")
        
            if self.batchsize_scheduler_bool:
                keys_batchsize_schedule = np.array(list(self.batchsize_schedule.keys()), int)
                if len(keys_batchsize_schedule[keys_batchsize_schedule < self.previously_trained_number_epoch]) != 0:
                    last_epoch_config = keys_batchsize_schedule[keys_batchsize_schedule < self.previously_trained_number_epoch].max()
                    self.reset_dataloader(self.batchsize_schedule[str(last_epoch_config)], self.previously_trained_number_epoch)
            
            self.JUNIPR_quark_model.train()
            self.JUNIPR_gluon_model.train()
            
            self.setup_loss()
            self.setup_optimiser(binary_case = True)
            
            if save_model_bool:
                self.quark_additional_parameters.save_configuration(os.path.join(self.result_path, "quark_model"))
                self.gluon_additional_parameters.save_configuration(os.path.join(self.result_path, "gluon_model"))
                  
            self.binary_train()
            
            if save_model_bool:
                self.JUNIPR_quark_model.save_model(os.path.join(self.result_path, "quark_model"))
                self.JUNIPR_gluon_model.save_model(os.path.join(self.result_path, "gluon_model"))
                
        if load_bool:
            # there should be a config file and the model parameters stored at that model_path
            model_path  = config.get(["Junipr_Model", "loading", "load_model_path"])
            config_path = config.get(["Junipr_Model", "loading", "load_model_config"])
            self.previously_trained_number_epoch = config.get(["Junipr_Model", "loading", "pre_trained_epochs"])
            self.previously_trained_number_steps = config.get(["Junipr_Model", "loading", "previously_trained_number_steps"])
            
            with open(config_path, 'r') as yaml_file:
                loaded_parameters = yaml.load(yaml_file, yaml.SafeLoader)
            additional_parameters = MainParameters(loaded_parameters)
            self.JUNIPR_model = JuniprNetwork(config=additional_parameters)
            self.JUNIPR_model.load_model(model_path)
            self.setup_loss()
            self.JUNIPR_model.train()
            self.setup_optimiser()
        
        if assess_bool and not(binary_junipr_bool):
            print("Assessing Unary Junipr")
            
            self.assess_unary_JUNIPR_proba_distributions()
            """
            self.JUNIPR_model.eval()
            if print_jets_bool:
                assess_number_of_jets  = config.get(["Junipr_Model", "assess_number_of_jets"])
                store_printed_jets = os.path.join(self.result_path, 'jets_printed/')
                os.makedirs(store_printed_jets, exist_ok=True)
            with torch.no_grad():
                output_loss = 0
                for batch_count, batch_input in enumerate(self.test_dataloader):
                    #batch_input = batch_input.to(self.device)
                    
                    batch_output = self.JUNIPR_model(batch_input)
                    
                    batch_loss = self.compute__batch_losses(batch_input, batch_output)
                    output_loss += batch_loss
                    if (print_jets_bool and batch_count < assess_number_of_jets):
                        _, event_probability_list = self.compute_losses(batch_input, batch_output, test_bool = True)
                        save_jets_as = os.path.join(store_printed_jets, "jet_" + str(batch_count))
                        self.log_jet_constructed(batch_input, event_probability_list, path = save_jets_as, tensorboard_logger_bool = False)
            
                    #if (batch_count % 1000 == 0):
                        #print("Assessing the model. Step {} | loss {}".format(batch_count, batch_loss))

                mean_loss = output_loss / len(self.test_dataloader)
            print("The mean probability computed is {}".format(mean_loss))
            """
            
        if assess_bool and binary_junipr_bool:
            print("Assessing Binary Junipr")
            self.JUNIPR_quark_model.eval()
            self.JUNIPR_gluon_model.eval()
            if print_jets_bool:
                assess_number_of_jets  = config.get(["Junipr_Model", "assess_number_of_jets"])
                store_printed_jets = os.path.join(self.result_path, 'jets_printed/')
                os.makedirs(store_printed_jets, exist_ok=True)
            with torch.no_grad():
                output_loss = 0
                output_accuracy = 0
                prediction_list = list()
                prediction_rounded_list = list()
                used_label_list = list()
                
                prediction_rounded_list_good_label = list()
                prediction_list_good_label =list()
                true_label_list_good_label = list()
                for batch_count, batch_input in enumerate(self.test_dataloader):
                    #batch_input = batch_input.to(self.device)

                    batch_output_quark = self.JUNIPR_quark_model(batch_input)
                    batch_output_gluon = self.JUNIPR_gluon_model(batch_input)
                    
                    #batch_loss, _, batch_accuracy = self.compute_binary_losses(batch_input, batch_output_quark, batch_output_gluon, test_loss = True)
                    
                    batch_loss, batch_accuracy = self.compute_batch_binary_losses(batch_input, batch_output_quark, batch_output_gluon, test_loss = True)
                    """
                    batch_loss2, _, accuracy_batch2 = self.compute_binary_losses(batch_input, batch_output_quark, batch_output_gluon, test_loss = True)
                    print("Batch loss with batch oriented {} and sequential {}".format(batch_loss, batch_loss2))
                    print("Batch accuracy with batch oriented {} and sequential {}".format(batch_accuracy, accuracy_batch2))
                    """
                    output_loss += batch_loss.item()
                    
                    prediction_label_proba, u_label, t_label = batch_accuracy
                    
                    prediction_label = round(prediction_label_proba)
                    prediction_list.append(prediction_label_proba)
                    prediction_rounded_list.append(prediction_label)
                    used_label_list.append(u_label)
  
                    if prediction_label == u_label:
                        output_accuracy += 1
                    
                    if not(t_label<0):
                        # it's a good label
                        prediction_list_good_label.append(prediction_label_proba)
                        true_label_list_good_label.append(t_label)
                        prediction_rounded_list_good_label.append(prediction_label)
                    
                    if (print_jets_bool and batch_count < assess_number_of_jets):
                        _, event_probability_list, _ = self.compute_batch_binary_losses(batch_input, batch_output_quark, batch_output_gluon, test_loss = True, detail_loss = True)
                        #_, event_probability_list2, _ = self.compute_binary_losses(batch_input, batch_output_quark, batch_output_gluon, test_loss = True, detail_loss = True)
                        
                        #save_jets_as = os.path.join(store_printed_jets, "jet_" + str(batch_count))
                        #self.log_jet_constructed(batch_input, event_probability_list, path = save_jets_as, tensorboard_logger_bool = False)
                        save_jets_as = os.path.join(store_printed_jets, "jet_" + str(batch_count))
                        self.log_jet_constructed(batch_input, event_probability_list, path = save_jets_as, tensorboard_logger_bool = False)
                        """
                        save_jets_as = os.path.join(store_printed_jets, "jet2_" + str(batch_count))
                        self.log_jet_constructed(batch_input, event_probability_list2, path = save_jets_as, tensorboard_logger_bool = False)
                        """
                        """
                        if (batch_count % 1000 == 0):
                            print("Assessing the model. Step {} | loss {}".format(batch_count, batch_loss))
                        """
                print("len(self.test_dataloader) ", len(self.test_dataloader))
                mean_loss = output_loss / len(self.test_dataloader)
                accuracy  = output_accuracy / len(self.test_dataloader)
                print("predicted labels: ", prediction_list_good_label)
                print("used_label_list ", used_label_list)
                print("true_label_list_good_label ", true_label_list_good_label)
                
                
                list_plot = [("Used labels", used_label_list, prediction_list)]
                list_plot_good = [("Good labels", true_label_list_good_label, prediction_list_good_label)]
                print("prediction_list_good_label: ", prediction_list_good_label)
                ROC_curve_plotter_from_values(list_plot, os.path.join(self.result_path, "roc_used_labels"))
                ROC_curve_plotter_from_values(list_plot_good, os.path.join(self.result_path, "roc_good_labels"))
                confusion_matrix_used = metrics.confusion_matrix(used_label_list, prediction_rounded_list)
                plot_confusion_matrix(cm = confusion_matrix_used, normalize=True)
                plt.savefig(os.path.join(self.result_path, 'confusion_matrix_normalised_used_labels.png'))
                plt.close()
                plot_confusion_matrix(cm = confusion_matrix_used, normalize=False)
                plt.savefig(os.path.join(self.result_path, 'confusion_matrix_used_labels.png'))
                plt.close()
                confusion_matrix_good= metrics.confusion_matrix(true_label_list_good_label, prediction_rounded_list_good_label)
                plot_confusion_matrix(cm = confusion_matrix_good, normalize=True)
                plt.savefig(os.path.join(self.result_path, 'confusion_matrix_normalised_good_labels.png'))
                plt.close()
                plot_confusion_matrix(cm = confusion_matrix_good, normalize=False)
                plt.savefig(os.path.join(self.result_path, 'confusion_matrix_good_labels.png'))
                plt.close()
            print("Analysing {} jets in total with {} with good labels".format(len(used_label_list), len(true_label_list_good_label)))
            print("The mean probability computed is {}".format(mean_loss))
            print("The accuracy computed is {}".format(accuracy))


            
