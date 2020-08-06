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
from sklearn import metrics

from tensorboardX import SummaryWriter
import tensorflow as tf

from DataLoaders import DataLoader_Set4
from .BaseRunner import _BaseRunner
from .Networks import JuniprNetwork
from Utils import tree_plotter, draw_tree, write_ROC_info, plot_confusion_matrix, ROC_curve_plotter_from_values, write_to_file
from Utils import MainParameters

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
        
            quark_additional_parameters = MainParameters(quark_loaded_parameters)
            gluon_additional_parameters = MainParameters(gluon_loaded_parameters)
            
            self.JUNIPR_quark_model = JuniprNetwork(config=quark_additional_parameters)
            self.JUNIPR_gluon_model = JuniprNetwork(config=gluon_additional_parameters)
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
    
        # Next parameter is used when further training a model (and modified in loading step of run), to restart at the last epoch trained (import for schedule).
        # The model will still be trained for the num_epochs given (it will be added to the current value so that the model will have globally trained
        # for previously_trained_number_epoch + self.num_epochs.
        self.previously_trained_number_epoch = 0

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
        self.train_dataset, self.validation_dataset, self.test_dataset = self.dataloader.load_separate_data()

        bool_suffle_test = False
        if self.binary_junipr and self.assess_bool:
            # Running a binary junipr and assessing it.
            # You have to shuffle the test in this case as otherwise you'll only draw the x first jets
            # which will be mostly quark if not only them.
            bool_suffle_test = True
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)#, num_workers=4)
        self.validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader  = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=bool_suffle_test)

    def reset_dataloader(self, new_batchsize, epoch):
        """
        This is used to change the batchsize of the train dataloader (to change before iterating over it)
        """
        print("\n#######################################################\n")
        print("Epoch {} | New batch size : {}".format(epoch, new_batchsize) )
        print("\n#######################################################\n")
        self.batch_size = new_batchsize
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=new_batchsize, shuffle=True)

    def setup_optimiser(self, binary_case = False):
        if self.optimiser_type == "adam":
            """
            # Not specifying them at the moment
            beta_1 = self.optimiser_params[0]
            beta_2 = self.optimiser_params[1]
            epsilon = self.optimiser_params[2]
            """
            if binary_case:
                self.optimiser = torch.optim.Adam(list(self.JUNIPR_quark_model.parameters()) + list(self.JUNIPR_gluon_model.parameters()), lr=self.lr)#,betas=(beta_1, beta_2),eps=epsilon)
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

        # Batch size  schedule
        self.batchsize_scheduler_bool = False
        if self.batch_scheduler == "junipr_paper_binary":
            self.batchsize_scheduler_bool = True
            self.batchsize_schedule = {"1": 10, "2": 100, "7": 1000, "17": 2000 }
        elif self.batch_scheduler == "junipr_unary_LONG":
            self.batchsize_scheduler_bool = True
                #self.batchsize_schedule = {"1": 10, "26": 100}
            self.batchsize_schedule = {"1": 2, "26": 4}

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
        
        if self.binary_junipr:
            # Again, the BCEWithLogitsLoss will perform the sigmoid itself
            self.binary_with_loss = True
            self.discriminator  = nn.BCEWithLogitsLoss(reduction = 'none')
            self.loss_modules["discriminator"] = self.discriminator

    def compute_losses(self, batch_input, batch_output, test_bool = False, binary_target = False):
        """
        To compute the loss in our batch-leaning environment: the sequences are flatten into an array.
        
        batch_input, batch_output are dictionnaries with the entries required.
        
        Note that the ignore_index specification from the nn.CrossEntropyLoss() is not sufficient here: padded values are passed through the networks.
        
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
   
        if test_bool:
            list_probabilities_per_jet = list()
        #element_input_very_ending = torch.tensor(np.ones((1,), dtype=int), requires_grad=False).long()
        element_input_very_ending = torch.FloatTensor([1])
        element_input_very_ending.detach()
        
        if binary_target:
            #batch_loss_tensor = torch.FloatTensor(np.zeros((input_n_branchings.size()[0], 1)), requires_grad=True)
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
            element_output_very_end   = output_very_end[0, batch_element, :] #there is a first component of size 1 that is due to the RNN structure (fixed to 1 here)
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
                    node_probability = p_mother_output_loss[node_counter] + p_end_output_loss[node_counter] +p_branch_z_output_loss[node_counter] + p_branch_t_output_loss[node_counter] + p_branch_p_output_loss[node_counter] + p_branch_d_output_loss[node_counter]
                
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
    
    def compute_binary_losses(self, batch_input, batch_output_quark, batch_output_gluon, test_loss = False, detail_loss = False):
        """
        To compute the binary loss in our batch-leaning environment: the sequences are flatten into an array.
          
        batch_input, batch_output are dictionnaries with the entries required.
          
        Note that the ignore_index specification from the nn.CrossEntropyLoss() is not sufficient here: padded values are passed through the networks.
          
        If test_bool is false, returns the batch loss.
        If test_bool is true, returns a batch_loss and a list of tuples (one tuple for each batch item): (item probability, list of probability at each node).
        
        This implementation assumes equal likelihood of having a quark or a gluon jet. It computes a sigmoid on log( Proba quark) - log (Proba gluon)
        
        Note that self.compute_losses returns probabilities as -log(Proba).
        """
        #start_loss = time.process_time()
        batch_loss = 0
        dataset_label = batch_input["dataset_label"] # a specific label that tells you whether the jet is from a quark (1) or gluon (0) rich table.
        true_label = batch_input["label"]   # the MC label telling you if it is a quark (1) gluon (0) or other (-1)
        size_batch = dataset_label.size()[0]
        """
        _, list_proba_quark = self.compute_losses(batch_input = batch_input, batch_output = batch_output_quark, test_bool = True)
        _, list_proba_gluon = self.compute_losses(batch_input = batch_input, batch_output = batch_output_gluon, test_bool = True)
        """
        
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
        else:
            prediction_label = torch.tensor(np.zeros((size_batch, 1)), requires_grad=False)
        
        detail_list_proba = list()
            #for indice in range(len(list_proba_quark)):
        correct_count = 0
        #start_loop_loss = time.process_time()
        for indice in range(size_batch):
            dataset_jet = dataset_label[indice].item()
            true_ID_jet = true_label[indice].item()
            used_label = true_ID_jet
            if true_ID_jet == -1:
                # how to deal with mis classified labels from the testing set ?
                # In this scenario, if such jets are present, consider the label to be the dataset enriched one (quark or gluon).
                used_label = dataset_jet
            used_label = torch.FloatTensor([used_label]).detach()
            
            if detail_loss:
                tuple_proba_quark = list_proba_quark[indice]
                tuple_proba_gluon = list_proba_gluon[indice]

                _, proba_quark_per_nodes = tuple_proba_quark
                _, proba_gluon_per_nodes = tuple_proba_gluon
            
            if test_loss:
                proba_quark = torch.tensor([tensor_proba_quark.item()])
                proba_gluon = torch.tensor([tensor_proba_gluon.item()])
                
            else:
                proba_quark = tensor_proba_quark[indice]
                proba_gluon = tensor_proba_gluon[indice]
            
            # The following is for supervised learning.
            #prediction_loss = self.loss_modules["discriminator"](-proba_quark + proba_gluon, used_label.long())
            
            # The following for semi-supervised learning: maximise the likelihood ratio itself, with the sign depending on the label
            # We use a sigmoid to make the loss smooth.
            if self.binary_with_loss:
                proba_ratio = - proba_quark + proba_gluon # we need to compute the real proba ration and use this on the discriminator.
                proba = self.loss_modules["discriminator"](proba_ratio, used_label)
            else:
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
                prediction = torch.sigmoid(-proba_quark + proba_gluon).item()
                prediction_label = prediction
                t_label = true_ID_jet
                u_label = used_label
            else:
                prediction = torch.round(torch.sigmoid(-proba_quark + proba_gluon)).item()
                prediction_label[indice] = prediction
                if prediction == used_label:
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
            accuracy = (prediction_label, u_label, t_label)
        else:
            # rough idea of what the accuracy is
            accuracy = correct_count / dataset_label.size()[0]

        return batch_loss, detail_list_proba, accuracy
                  
                  
    def train(self):
        """
        The training loop. Note that manipulation of the input/output into the losses is very particular here.
        """
        # Set model to train mode
        self.JUNIPR_model.train()
        step_count = 0
        
        # This part sets up some logging information for later plots.
        if self.logger_data_bool:
            # initiate the logger
            write_to_file(os.path.join(self.result_path, "logger_info.txt"), ["#step", "train_loss", "test_loss"], action = 'w')
            write_to_file(os.path.join(self.result_path, "saved_info.txt"), ["#epoch", "step"], action = 'w')
                
        time_start_sequence = time.process_time()
        #torch.backends.cudnn.benchmark = True
        last_epoch_base_5 = str(0)
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
            """
            # More general version that would work even if retraining mid range (so if at epoch 14, would know to take lr of epoch 11). For retraining purposes
            if self.lr_scheduler_bool:
                closest_epoch_base_5 = str((epoch - epoch % 5)+1)
                if closest_epoch_base_5 != last_epoch_base_5:
                    last_epoch_base_5 = closest_epoch_base_5
                    if closest_epoch_base_5 in self.lr_schedule:        # don't do anything if you leave the range, hence the else.
                        self.lr = self.lr_schedule[closest_epoch_base_5]
                        print("\n#######################################################\n")
                        print("Epoch {} | New learning rate : {}".format(epoch, self.lr))
                        print("\n#######################################################\n")
                        # Update the learning rate parameter of the optimiser.
                        for param_group in self.optimiser.param_groups:
                            param_group['lr'] = self.lr
                    else:
                        # you've reached the end of the range of lr_scheduler, deactive the modifier
                        self.lr_scheduler_bool = False
            """
            
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
                
                batch_output = self.JUNIPR_model(batch_input)
                #to_loss_time = time.process_time()
                #print("Time to reach loss step {}".format(to_loss_time - very_start))
                batch_loss = self.compute_losses(batch_input, batch_output) / size_batch
                #loss_time = time.process_time()
                #print("Time to compute loss step {}".format(loss_time - to_loss_time))
                epoch_loss += float(batch_loss * size_batch)
                
                self.optimiser.zero_grad()
                batch_loss.backward()
                self.optimiser.step()
                #print("Time to backprop loss step {}".format(time.process_time() - loss_time))
                #print("Finished step {} |Time for whole step {}".format(step_count, time.process_time() - very_start))
                if (int(step_count* self.batch_size) % int(self.test_frequency * 200) == 0):
                    print("Training {} step | batch loss : {}".format(step_count, float(batch_loss)))
                    # Report result to TensorBoard
                    self.writer.add_scalar("training_loss", float(batch_loss), step_count)
                    test_loss = self.test_loop(step=step_count)
                    time_end_sequence = time.process_time()
                    print("Time it took for the sequence: {}".format(time_end_sequence - time_start_sequence))
                    if self.logger_data_bool:
                        # append to the logger
                        write_to_file(os.path.join(self.result_path, "logger_info.txt"), [step_count, float(batch_loss), test_loss], action = 'a', limit_decimal = True)
                        
                    self.JUNIPR_model.train()
                    time_start_sequence = time.process_time()
                #time_start_load = time.process_time()
            if (epoch % 5 == 0 and self.save_model_bool):
                self.JUNIPR_model.save_model(self.result_path)
                if self.logger_data_bool:
                    # append to the logger to confirm the last updates
                    write_to_file(os.path.join(self.result_path, "saved_info.txt"), [epoch, step_count], action = 'a')
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
        
        # This part sets up some logging information for later plots.
        if self.logger_data_bool:
            # initiate the logger
            write_to_file(os.path.join(self.result_path, "logger_info.txt"), ["#step", "train_loss", "test_loss", "train_acc", "test_acc"], action = 'w')
            write_to_file(os.path.join(self.result_path, "saved_info.txt"), ["#epoch", "step"], action = 'w')
        
        torch.backends.cudnn.benchmark = True
        time_start_sequence = time.process_time()
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
                #very_start = time.process_time()
                #print("Starting step {}".format(step_count))
                step_count += 1
                batch_count += 1
                
                input_n_branchings = batch_input["n_branchings"]
                size_batch = input_n_branchings.size()[0]
                raise ValueError("end")
                batch_output_quark = self.JUNIPR_quark_model(batch_input)
                #print("Going in gluon")
                batch_output_gluon = self.JUNIPR_gluon_model(batch_input)
                #print("Going in loss")
                #to_loss_time = time.process_time()
                #print("Time to reach loss step {}".format(to_loss_time - very_start))
                
                batch_loss, _, accuracy_batch = self.compute_binary_losses(batch_input, batch_output_quark, batch_output_gluon)
                #batch_loss = batch_loss / size_batch
                #loss_time = time.process_time()
                #print("Time to compute loss step {}".format(loss_time - to_loss_time))
                
                batch_loss = batch_loss / size_batch
                self.optimiser.zero_grad()
                batch_loss.backward()
                self.optimiser.step()
    
                epoch_loss += float(batch_loss)
                
                #print("Time to finish backprop loss step {}".format(time.process_time() - loss_time))
                #print("Finished step {} |Time for whole step {}".format(step_count, time.process_time() - very_start))
                
                #print("Training {} step | batch loss : {} | accuracy : {}".format(step_count, batch_loss, accuracy_batch))

                if (step_count % self.test_frequency == 0):
                    print("Training {} step | batch loss : {} | accuracy : {}".format(step_count, float(batch_loss), float(accuracy_batch)))
                    # Report result to TensorBoard
                    self.writer.add_scalar("training_loss", float(batch_loss), step_count)
                    test_loss, test_acc = self.binary_test_loop(step=step_count)
                    time_end_sequence = time.process_time()
                    print("Time it took for the sequence: {}".format(time_end_sequence - time_start_sequence))
                    if self.logger_data_bool:
                        # append to the logger
                        write_to_file(os.path.join(self.result_path, "logger_info.txt"), [step_count, float(batch_loss), test_loss, float(accuracy_batch), test_acc], action = 'a', limit_decimal = True)
                    self.JUNIPR_quark_model.train()
                    self.JUNIPR_gluon_model.train()
                    time_start_sequence = time.process_time()
                #time_start_load = time.process_time()
            if (epoch % 1 == 0 and self.save_model_bool):
                self.JUNIPR_quark_model.save_model(os.path.join(self.result_path, "quark_model"))
                self.JUNIPR_gluon_model.save_model(os.path.join(self.result_path, "gluon_model"))
                if self.logger_data_bool:
                    # append to the logger to confirm the last updates
                    write_to_file(os.path.join(self.result_path, "saved_info.txt"), [epoch, step_count], action = 'a')
            print("\n#######################################################\n")
            print("Epoch {} | loss : {}".format(epoch, epoch_loss/batch_count))
            print("\n#######################################################\n")
                  
    def binary_test_loop(self, step:int):
        self.JUNIPR_quark_model.eval()
        self.JUNIPR_gluon_model.eval()
        """
        get_an_event = True
        size_sample = len(self.test_dataloader)
        index = random.randint(0, size_sample - 1)
        """
        with torch.no_grad():
            output_loss = 0
            output_accuracy = 0
            total_size = 0
            for batch_count, batch_input in enumerate(self.validation_dataloader):
                #batch_input = batch_input.to(self.device)
                input_n_branchings = batch_input["n_branchings"]
                size_batch = input_n_branchings.size()[0]
                total_size += size_batch
                
                batch_output_quark = self.JUNIPR_quark_model(batch_input)
                batch_output_gluon = self.JUNIPR_gluon_model(batch_input)
                
                batch_loss, _, accuracy = self.compute_binary_losses(batch_input, batch_output_quark, batch_output_gluon)
                output_loss += batch_loss
                output_accuracy += accuracy * size_batch
                """
                if batch_count == index:
                    _, event_probability_list = self.compute_binary_losses(batch_input, batch_output_quark, batch_output_gluon)
                    self.log_jet_constructed(batch_input, event_probability_list, step = step)
                    #get_an_event = False
                """
            mean_loss = output_loss / total_size
            mean_accuracy = output_accuracy / total_size
            
            print("Validation {} step | mean loss = {} | mean accuracy = {}".format(step, float(mean_loss), float(mean_accuracy)))
            self.writer.add_scalar("validation_loss", float(mean_loss), step)
        return mean_loss, mean_accuracy

    def test_loop(self, step:int):
        self.JUNIPR_model.eval()
        """
        get_an_event = True
        size_sample = len(self.test_dataloader)
        index = random.randint(0, size_sample - 1)
        """
        with torch.no_grad():
            output_loss = 0
            for batch_count, batch_input in enumerate(self.validation_dataloader):
                  #batch_input = batch_input.to(self.device)
                  input_n_branchings = batch_input["n_branchings"]
                  size_batch = input_n_branchings.size()[0]
          
                  batch_output = self.JUNIPR_model(batch_input)
          
                  batch_loss = self.compute_losses(batch_input, batch_output) / size_batch
                  output_loss += batch_loss
                  """
                  if batch_count == index:
                    _, event_probability_list = self.compute_losses(batch_input, batch_output, test_bool = True)
                    self.log_jet_constructed(batch_input, event_probability_list, step = step)
                    #get_an_event = False
                 """
            mean_loss = output_loss / len(self.validation_dataloader)
                  
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


    def run(self, config, train_bool = False, load_bool = False, assess_bool = False, print_jets_bool = False, save_model_bool = False, binary_junipr_bool = False):
        """
        The centre of the Junipr Runner. Runs the operations required by the configuration
        """
        if train_bool and not(binary_junipr_bool):
            # training a single version of junipr
            self.train()
            if save_model_bool:
                  self.JUNIPR_model.save_model(self.result_path)
                  
        elif train_bool and binary_junipr_bool:
            self.JUNIPR_quark_model.train()
            self.JUNIPR_gluon_model.train()
            
            self.setup_loss()
            self.setup_optimiser(binary_case = True)
            
            if save_model_bool:
                quark_additional_parameters.save_configuration(os.path.join(self.result_path, "quark_model"))
                gluon_additional_parameters.save_configuration(os.path.join(self.result_path, "gluon_model"))
                  
            self.binary_train()
            
            if save_model_bool:
                self.JUNIPR_quark_model.save_model(os.path.join(self.result_path, "quark_model"))
                self.JUNIPR_gluon_model.save_model(os.path.join(self.result_path, "gluon_model"))
                
        if load_bool:
            # there should be a config file and the model parameters stored at that model_path
            model_path  = config.get(["Junipr_Model", "loading", "load_model_path"])
            config_path = config.get(["Junipr_Model", "loading", "load_model_config"])
            self.previously_trained_number_epoch = config.get(["Junipr_Model", "loading", "pre_trained_epochs"])
            
            if self.lr_scheduler_bool:
                keys_lr_schedule = np.array(list(self.lr_schedule.keys()), int)
                if len(keys_lr_schedule[keys_lr_schedule < self.previously_trained_number_epoch]) != 0:
                    last_epoch_config = keys_lr_schedule[keys_lr_schedule < self.previously_trained_number_epoch].max()
                    self.lr = self.lr_schedule[str(last_epoch_config)]
      
            if self.batchsize_scheduler_bool:
                keys_batchsize_schedule = np.array(list(self.batchsize_schedule.keys()), int)
                if len(keys_batchsize_schedule[keys_batchsize_schedule < self.previously_trained_number_epoch]) != 0:
                    last_epoch_config = keys_batchsize_schedule[keys_batchsize_schedule < self.previously_trained_number_epoch].max()
                    self.lr = self.batchsize_schedule[str(last_epoch_config)]
            
            # If need be, you have to find the last configuration in the batchsize and learning rates schedules and use these ones.
            
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
                    
                    batch_loss = self.compute_losses(batch_input, batch_output)
                    output_loss += batch_loss
                    if (print_jets_bool and batch_count < assess_number_of_jets):
                        _, event_probability_list = self.compute_losses(batch_input, batch_output, test_bool = True)
                        save_jets_as = os.path.join(store_printed_jets, "jet_" + str(batch_count))
                        self.log_jet_constructed(batch_input, event_probability_list, path = save_jets_as, tensorboard_logger_bool = False)
                    """
                    if (batch_count % 1000 == 0):
                        print("Assessing the model. Step {} | loss {}".format(batch_count, batch_loss))
                    """
                mean_loss = output_loss / len(self.test_dataloader)
            print("The mean probability computed is {}".format(mean_loss))
            
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
                    
                    batch_loss, _, batch_accuracy = self.compute_binary_losses(batch_input, batch_output_quark, batch_output_gluon, test_loss = True)
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
                        _, event_probability_list, _ = self.compute_binary_losses(batch_input, batch_output_quark, batch_output_gluon, test_loss = True, detail_loss = True)
                        save_jets_as = os.path.join(store_printed_jets, "jet_" + str(batch_count))
                        self.log_jet_constructed(batch_input, event_probability_list, path = save_jets_as, tensorboard_logger_bool = False)
                        """
                        if (batch_count % 1000 == 0):
                            print("Assessing the model. Step {} | loss {}".format(batch_count, batch_loss))
                        """
                mean_loss = output_loss / len(self.test_dataloader)
                accuracy  = output_accuracy / len(self.test_dataloader)
                list_plot = [("Used labels", used_label_list, prediction_list)]
                list_plot_good = [("Good labels", true_label_list_good_label, prediction_list_good_label)]
                
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


            
