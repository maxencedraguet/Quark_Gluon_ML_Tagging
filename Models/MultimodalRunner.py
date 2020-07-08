#############################################################################
#
# MultimodalRunner.py
#
# A runner loading mutliple fed models and some specific data to assess them.
#
# Author -- Maxence Draguet (19/05/2020)
#
#############################################################################

from abc import ABC, abstractmethod
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
import joblib as jl

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

from DataLoaders import DataLoader_Set1, DataLoader_Set2
from .BaseRunner import _BaseRunner
from .BDTRunner import BDTRunner
from .Networks import NeuralNetwork
from Utils import write_ROC_info, plot_confusion_matrix, ROC_curve_plotter_from_values
from Utils import MainParameters

class MultimodalRunner(_BaseRunner):
    def __init__(self, config: Dict) -> None:
        self.extract_parameters(config)
        self.load_models()
        self.setup_dataset(config)
        self.assess_models()

    def extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        """
        self.experiment_timestamp = config.get("experiment_timestamp")
        self.absolute_data_path = config.get(["absolute_data_path"])
        self.result_path = config.get(["log_path"])
        os.makedirs(self.result_path, exist_ok=True)
        self.dataset = config.get(["dataset"])
        
        #Test_set_type: either
        #   - regular (proportional to all data): regular
        #   - cross-section weighted:             cs            Only Set2
        #   - specific energy range:              energy        Only Set2
        #   - specific file key (process):        process       Only Set2
        
        self.test_set_type = config.get(["Multi_model", "data", "test_set_type"])
        self.energy_range = config.get(["Multi_model", "data", "energy_range"])
        self.process_list = config.get(["Multi_model", "data", "process_list"])
        
        # expecting a list of tuples:
        # (model_type, path_to_model, path_to_model_config.yaml, description_string)
        self.model_folder = config.get(["Multi_model", "models"])
        if self.model_folder == "store_model":
            from store_model import models_to_load
        #self.list_model_loader = models_to_load.models_to_load_list
        elif self.model_folder == "store_model2":
            from store_model2 import models_to_load
#self.list_model_loader = models_to_load.models_to_load_list
        else:
            raise ValueError("Model folder {} not recognised". format(self.model_folder))
        self.list_model_loader = models_to_load.models_to_load_list # expecting a list of tuples:
        self.need_analyse_given_BDT = True

    def load_models(self):
        """
        Loads the model from self.list_model_loader indication and store them in
        list of models based on type.
        """
        self.NN_models = list()
        self.BDT_models = list()
        for model_type, model_path, config_yaml, description in self.list_model_loader:
            print(model_type, model_path, config_yaml, description)
            if model_type == "NN":
                network = self.load_NN_model(path = model_path, model_config = config_yaml)
                self.NN_models.append( (network, model_path, description) )

            elif model_type == "BDT":
                self.BDT_models.append( (jl.load(model_path), model_path, description) )

            else:
                raise ValueError("Model type {} not recognised". format(model_type))

    def load_NN_model(self, path, model_config):
        """
        Routine to load a NN model with PyTorch.
        Creates a NNetwork object, sets the right parameter as
        loaded from the associated config file (in config_yaml)
        and return the NNetwork hencer formed
        """
        with open(model_config, 'r') as yaml_file:
            loaded_parameters = yaml.load(yaml_file, yaml.SafeLoader)
        additional_parameters = MainParameters(loaded_parameters)
        network = NeuralNetwork(source = "NN_Model", config=additional_parameters)
        network.load_model(path)
        network.eval()
        return network

    def setup_dataset(self, config: Dict):
        if self.test_set_type == "regular":
            if self.dataset == "Set1":
                self.dataloader = DataLoader_Set1(config)
            if self.dataset == "Set2":
                self.dataloader = DataLoader_Set2(config)
            self.data = self.dataloader.load_separate_data()

        elif self.test_set_type == "cs":
            pass
        elif self.test_set_type == "energy":
            pass
        elif self.test_set_type == "process":
            pass
        else:
            raise ValueError("Test set type {} not recognised". format(self.test_set_type))

        # Formatting for PyTorch
        self.tensor_data = dict()
        self.tensor_data["input_test"]   = torch.Tensor(self.data["input_test"])
        self.tensor_data["output_test"]  = torch.Tensor(self.data["output_test"])
        self.test_dataset = torch.utils.data.TensorDataset(self.tensor_data["input_test"] , self.tensor_data["output_test"])
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False)

    def assess_models(self):
        """
        For models loaded, runs each of them on the dataset selected.
        The results are stored in self.result_list and in text files.
        Finally, a ROC figure with all info is plotted.
        """
        self.result_list = list()
        for NN_model, model_path, description in self.NN_models:
            NN_model.eval()
            self.run_test_NN(NN_model, model_path, description)
        for BDT_model, model_path, description in self.BDT_models:
            self.run_test_BDT(BDT_model, model_path, description)

        ROC_curve_plotter_from_values(self.result_list, self.result_path)

    def run_test_NN(self, NN_model, path, description):
        """
        Run a NN model on the test set.
        """
        y_pred_proba_list = []
        y_pred_list = []
        y_true_list = []
        with torch.no_grad():
            output_acc  = 0
            for batch_input, batch_target in self.test_dataloader:
                #batch_input = batch_input.to(self.device)
                NN_output = NN_model(batch_input)
                batch_prediction = torch.sigmoid(NN_output)
                y_pred_proba_list.append(batch_prediction.numpy())
                batch_prediction = torch.round(batch_prediction)
                y_pred_list.append(batch_prediction.numpy())
                y_true_list.append(batch_target.numpy())
   
                output_acc  += self.compute_accuracy(batch_prediction, batch_target.unsqueeze(1)) * len(batch_input)

        mean_accuracy = output_acc  / len(self.test_dataloader)
                
        print("Model {} | accuracy = {}".format(description, float(mean_accuracy)))
        #Store the results into a fixed format to run a final common assessment.
        
        pred_proba = [a.squeeze().tolist() for a in y_pred_proba_list]
        pred_label = [a.squeeze().tolist() for a in y_pred_list]
        true_label = [a.squeeze().tolist() for a in y_true_list]
        self.assess(pred_proba, pred_label, true_label, description)

    def run_test_BDT(self, BDT_model, path, description):
        """
        Run a BDT model on the test set.
        """
        pred_label= BDT_model.predict(self.data["input_test"])
        pred_proba = BDT_model.predict_proba(self.data["input_test"])
        accuracy = metrics.accuracy_score(self.data["output_test"], pred_label)

        print("Model {} | accuracy = {}".format(description, float(accuracy)))

        self.assess(pred_proba[:,1], pred_label, self.data["output_test"], description)

    def assess(self, pred_proba, pred_label, true_label, model_indicator):
        """
        Runs some statistics gathering methods, storing results in the result_path given.
        """
        model_name = model_indicator.replace("_", " ")
        print(metrics.classification_report(true_label, pred_label))
        self.confusion_matrix = metrics.confusion_matrix(true_label, pred_label)
        plot_confusion_matrix(cm = self.confusion_matrix, normalize=True, title = "Confusion Matrix for Model: " + model_name )
        plt.savefig(os.path.join(self.result_path, 'confusion_matrix'+model_indicator+'.png'))
        write_ROC_info(os.path.join(self.result_path, 'ROC_info_'+model_indicator+ '.txt'),
                        true_label, pred_proba)
        self.result_list.append( (model_name, true_label, pred_proba) )
        
        # Take strictly once (first model assessed) the result of the given BDT in data
        if self.need_analyse_given_BDT:
            self.need_analyse_given_BDT = False
            write_ROC_info(os.path.join(self.result_path, 'ROC_info_given_BDT.txt'),
                        true_label, self.data["output_BDT_test"])
            self.result_list.append( ("Given BDT", true_label, self.data["output_BDT_test"]) )

    def compute_accuracy(self, y_pred, y_test):
        """
        Compute accuracy for pytorh list
        """
        correct_results_sum = (y_pred == y_test).sum().float()
        accuracy = torch.round(correct_results_sum/y_test.shape[0] *100)
        return accuracy
