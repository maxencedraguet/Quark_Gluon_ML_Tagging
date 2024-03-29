#############################################################################
#
# NNRunner.py
#
# A neural network runner using PyTorch to implement the NNetwork.py model
#
# Author -- Maxence Draguet (1/06/2020)
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
from torch.utils.data import random_split
from sklearn import metrics

from tensorboardX import SummaryWriter

from DataLoaders import DataLoader_Set1, DataLoader_Set2
from .BaseRunner import _BaseRunner
from .Networks import NeuralNetwork
from Utils import write_ROC_info, plot_confusion_matrix, ROC_curve_plotter_from_values, write_to_file

class NNRunner(_BaseRunner):

    def __init__(self, config: Dict):
        self.extract_parameters(config)
        self.setup_NN(config)
        self.setup_optimiser()
        self.setup_dataloader(config)
        self.writer = SummaryWriter(self.result_path) # A tensorboard writer
        self.run()
    
    def extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        """
        self.experiment_timestamp = config.get("experiment_timestamp")
        self.absolute_data_path = config.get(["absolute_data_path"])
        self.result_path = config.get(["log_path"])
        os.makedirs(self.result_path, exist_ok=True)
        self.dataset = config.get(["dataset"])
        self.save_model_bool = config.get(["save_model"])
        self.seed = config.get(["seed"])
        self.logger_data_bool = config.get(["logger_data"])
        
        # dimensions should be a list of successive layers.
        self.lr = config.get(["NN_Model", "lr"])
        self.lr_scheduler = config.get(["NN_Model", "lr_scheduler"])
        self.num_epochs = config.get(["NN_Model", "epoch"])
        self.test_frequency = config.get(["NN_Model", "test_frequency"])
        self.optimiser_type = config.get(["NN_Model", "optimiser", "type"])
        self.optimiser_params = config.get(["NN_Model", "optimiser", "params"])
        self.weight_decay = config.get(["NN_Model", "optimiser", "weight_decay"])
        self.batch_size = config.get(["NN_Model", "batch_size"])
        self.loss_function = config.get(["NN_Model", "loss_function"])
        self.validation_size = config.get(["NN_Model", "validation_size"])
        self.extract_BDT_info = config.get(["NN_Model", "extract_BDT_info"])

    def setup_NN(self, config: Dict):
        #if self.network_type = "neural_network":
        self.network = NeuralNetwork(source = "NN_Model", config=config)
        self.last_non_lin = self.network.get_last_non_linearity()
        self.setup_loss()

    def setup_dataloader(self, config: Dict)->None:
        """
        Set up the dataloader for PyTorch execution
        """
        self.setup_dataset(config) # mother class function.
        self.data["input_train"]  = torch.Tensor(self.data["input_train"])
        self.data["input_test"]   = torch.Tensor(self.data["input_test"])
        self.data["output_train"] = torch.Tensor(self.data["output_train"])
        self.data["output_test"]  = torch.Tensor(self.data["output_test"])
        # Train
        self.train_dataset = torch.utils.data.TensorDataset(self.data["input_train"], self.data["output_train"])
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        # Test
        self.test_dataset = torch.utils.data.TensorDataset(self.data["input_test"] , self.data["output_test"])
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False)
    
        # Isolate a portion of the test set for visualising performance in training:
        test_dataset_size_used = int(len(self.test_dataset))
        val_size = int(test_dataset_size_used * self.validation_size)
        test_size = test_dataset_size_used - val_size
        self.validation_set, _ = random_split(self.test_dataset, [val_size, test_size])
        self.validation_dataloader = torch.utils.data.DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=True)
        
    def setup_optimiser(self):
        if self.optimiser_type == "adam":
            beta_1 = self.optimiser_params[0]
            beta_2 = self.optimiser_params[1]
            epsilon = self.optimiser_params[2]
            self.optimiser = torch.optim.Adam(self.network.parameters(), lr=self.lr,
                                              betas=(beta_1, beta_2),eps=epsilon,
                                              weight_decay= self.weight_decay)
        else:
            raise ValueError("Optimiser {} not recognised". format(self.optimiser_type))

    def setup_loss(self):
        if self.loss_function == "BCE_log": # binary cross entropy loss with logits (includes a sigmoid transform)
            self.loss = nn.BCEWithLogitsLoss()
        elif self.loss_function == "BCE":   # binary cross entropy loss with logits (without a sigmoid transform)
            self.loss = nn.BCELoss()
        else:
            raise ValueError("Loss {} not recognised". format(self.loss_function))

    def train(self):
        # Set model to train mode
        self.network.train()
        # This part sets up some logging information for later plots.
        if self.logger_data_bool:
            # Initiate the loggers.
            write_to_file(os.path.join(self.result_path, "logger_info.txt"), ["#step", "train_loss", "test_loss", "train_acc", "test_acc"], action = 'w')
            write_to_file(os.path.join(self.result_path, "saved_info.txt"), ["#epoch", "step"], action = 'w')
            write_to_file(os.path.join(self.result_path, "logger_epoch_info.txt"), ["#epoch", "train_loss", "train_acc"], action = 'w')

        step_count = 0
        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0
            epoch_acc  = 0
            sample_count = 0
            # If a learning rate schedule is used:
            if self.lr_scheduler:
                if (epoch == int((self.num_epochs + 1)/3) or epoch == int((self.num_epochs + 1)/3)*2):
                    self.lr *= 0.5
                    print("Learning Rate schedule: ", self.lr)
                    # Update the learning rate parameter of the optimiser.
                    for param_group in self.optimiser.param_groups:
                        param_group['lr'] = self.lr

            for batch_input, batch_target in self.train_dataloader:
                """
                if self.log_to_df:
                    self.logger_df.append(pd.Series(name=step_count))
                """
                # Move the batch input onto the GPU if necessary.
                #batch_input = batch_input.to(self.device)
                step_count += 1
                size_batch = batch_input.size()[0]
                sample_count += size_batch
                
                NN_output = self.network(batch_input)
                y_pred_tag = torch.round(torch.sigmoid(NN_output)).detach()
                
                output_loss = self.loss(NN_output, batch_target.unsqueeze(1))
                output_acc  = self.compute_accuracy(y_pred_tag, batch_target.unsqueeze(1))
                epoch_loss += (output_loss.item() * size_batch)
                epoch_acc  += (output_acc.item() * size_batch)
                 
                self.optimiser.zero_grad()
                output_loss.backward()
                self.optimiser.step()
            
                # Report result to TensorBoard
                self.writer.add_scalar("training_loss", float(output_loss), step_count)
                self.writer.add_scalar("training_acc",  float(output_acc),  step_count)
                #print("Training {} step | loss =  {} and accuracy = {}".format(step_count, float(output_loss), float(output_acc)))
                
                if (step_count% self.test_frequency == 0):
                    print("Training {} step | loss =  {} and accuracy = {}".format(step_count, float(output_loss), float(output_acc)))
                    test_loss, test_acc = self.test_loop(step=step_count)
                    self.network.train()
                    if self.logger_data_bool:
                        write_to_file(os.path.join(self.result_path, "logger_info.txt"), [int(step_count), float(output_loss), float(test_loss), float(output_acc), float(test_acc)], action = 'a', limit_decimal = True)
                    
            print("Training {} epochs | loss =  {} and accuracy = {}".format(epoch, float(epoch_loss/sample_count), float(epoch_acc/sample_count)))
            if (epoch % 1 == 0 and self.save_model_bool):
                self.network.save_model(self.result_path)
                if self.logger_data_bool:
                    # append to the logger to confirm the last updates
                    write_to_file(os.path.join(self.result_path, "saved_info.txt"), [epoch, step_count], action = 'a')
            if self.logger_data_bool:
                write_to_file(os.path.join(self.result_path, "logger_epoch_info.txt"), [epoch, float(epoch_loss/sample_count), float(epoch_acc/sample_count)], action = 'a')

    def test_loop(self, step:int):
        #print("Start testing loop")
        self.network.eval()
        with torch.no_grad():
            output_loss = 0
            output_acc  = 0
            sample_count = 0
            for batch_input, batch_target in self.validation_dataloader:
                #batch_input = batch_input.to(self.device)
                size_batch = batch_input.size()[0]
                sample_count += size_batch
                
                NN_output = self.network(batch_input)
                batch_prediction = torch.round(torch.sigmoid(NN_output)).detach()
                
                output_loss += float(self.loss(NN_output, batch_target.unsqueeze(1)) * size_batch)
                output_acc  += float(self.compute_accuracy(batch_prediction, batch_target.unsqueeze(1)) * size_batch)

            mean_loss     = output_loss / sample_count
            mean_accuracy = output_acc  / sample_count
            
            print("testing {} step | loss =  {} and accuracy = {}".format(step, float(mean_loss), float(mean_accuracy)))
            self.writer.add_scalar("test_loss", float(mean_loss),     step)
            self.writer.add_scalar("test_acc",  float(mean_accuracy), step)
        return mean_loss, mean_accuracy

    def test(self):
        self.network.eval()
        y_pred_proba_list = []
        y_pred_list = []
        y_true_list = []
        with torch.no_grad():
            output_loss = 0
            output_acc  = 0
            for batch_input, batch_target in self.test_dataloader:
                #batch_input = batch_input.to(self.device)
                NN_output = self.network(batch_input)
                batch_prediction = torch.round(torch.sigmoid(NN_output))
                output_loss += self.loss(NN_output, batch_target.unsqueeze(1)) * len(batch_input)
                output_acc  += self.compute_accuracy(batch_prediction, batch_target.unsqueeze(1)) * len(batch_input)
                
                y_pred_proba_list.append(torch.sigmoid(NN_output).numpy())
                y_pred_list.append(batch_prediction.numpy())
                y_true_list.append(batch_target.numpy())
            mean_loss     = output_loss / len(self.test_dataloader)
            mean_accuracy = output_acc  / len(self.test_dataloader)
            
        print("Final testing | loss =  {} and accuracy = {}".format(float(mean_loss), float(mean_accuracy)))
        
        self.data["test_predictions_proba"] = [a.squeeze().tolist() for a in y_pred_proba_list]
        self.data["test_predictions"] = [a.squeeze().tolist() for a in y_pred_list]
        self.data["output_test"]      = [a.squeeze().tolist() for a in y_true_list]
        self.assess()
                
    def assess(self):
        """
        Runs some statistics gathering methods, storing results in the result_path given. 
        """
        print(metrics.classification_report(self.data["output_test"], self.data["test_predictions"]))
        self.confusion_matrix = metrics.confusion_matrix(self.data["output_test"], self.data["test_predictions"])
        plot_confusion_matrix(cm = self.confusion_matrix, normalize=True)
        plt.savefig(os.path.join(self.result_path, 'confusion_matrix_normalised.png'))
        write_ROC_info(os.path.join(self.result_path, 'test_label_pred_proba_NN.txt'),
                       self.data["output_test"], self.data["test_predictions_proba"])
                       
        list_plot= [("NN", self.data["output_test"], self.data["test_predictions_proba"])]
                    
        if self.extract_BDT_info:
            write_ROC_info(os.path.join(self.result_path, 'test_label_pred_proba_given_BDT.txt'),
                           self.data["output_test"], self.data["output_BDT_test"]["BDTScore"])
        
            list_plot.append( ("BDT", self.data["output_test"], self.data["output_BDT_test"]["BDTScore"]) )
        
        ROC_curve_plotter_from_values(list_plot, self.result_path)
    
    def compute_accuracy(self, y_pred, y_test):
        correct_results_sum = (y_pred == y_test).sum().float()
        accuracy = correct_results_sum/y_test.shape[0]
        return accuracy

    def run(self):
        self.train()
        self.test()
        if self.save_model_bool:
            self.network.save_model(self.result_path)
    
