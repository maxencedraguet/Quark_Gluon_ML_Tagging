#############################################################################
#
# BDTRunner.py
#
# A BDT runner making a BDT model using sklearn.
#
# Author -- Maxence Draguet (19/05/2020)
#
#############################################################################

from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
from typing import Dict
from joblib import dump, load

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

from DataLoaders import DataLoader_Set1
from .BaseRunner import _BaseRunner

class BDTRunner(_BaseRunner):
    
    def __init__(self, config: Dict):
        
        self._extract_parameters(config)
        self.classifier = AdaBoostClassifier(n_estimators= self.n_estim,
                                        base_estimator= self.base_estimator,
                                        learning_rate= self.lr )
        self._setup_dataset(config)
        """
        if self.grid_search_bool:
            self.run_grid_search()
        else:
            self.run()
        """
    def _extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        """
        self.experiment_timestamp = config.get("experiment_timestamp")
        self.relative_data_path = config.get(["relative_data_path"])
        self.result_path = config.get(["log_path"])
        os.makedirs(self.result_path, exist_ok=True)
        self.dataset = config.get(["dataset"])
        self.save_model_bool = config.get(["save_model"])
        
        self.grid_search_bool = config.get(["BDT_model", "grid_search"])
        self.n_estim = config.get(["BDT_model", "n_estim"])
        self.base_estimator = config.get(["BDT_model", "base_estimator"])
        self.max_depth = config.get(["BDT_model", "max_depth"])
        self.lr = config.get(["BDT_model", "lr"])
        
        if self.base_estimator == "DecisionTreeClassifier" and self.max_depth:
            self.base_estimator = DecisionTreeClassifier(max_depth = self.max_depth)

    def checkpoint_df(self, step: int) -> None:
        raise NotImplementedError("Base class method")
    
    def _setup_dataset(self, config: Dict):
        if self.dataset == "Set1":
            self.dataloader = DataLoader_Set1(config)
            self.data = self.dataloader.load_separate_data()
    
    def train(self):
        self.model = self.classifier.fit(self.data["input_train"], self.data["output_train"])
        self.data["training_output_predictions"] = self.model.predict(self.data["input_train"])
        self.training_accuracy = metrics.accuracy_score(self.data["output_train"], self.data["training_output_predictions"])
    
    def predict(self):
        self.data["output_predictions"] = self.model.predict(self.data["input_test"])
        self.accuracy = metrics.accuracy_score(self.data["output_test"], self.data["output_predictions"])
        self.precision = metrics.average_precision_score(self.data["output_test"], self.data["output_predictions"])
        self.confusion_matrix = metrics.confusion_matrix(self.data["output_test"], self.data["output_predictions"])
        print("The training accuracy obtained is ", self.training_accuracy)
        print("The test accuracy obtained is ", self.accuracy)
        print("The test precision obtained is ", self.precision)

    def run(self):
        print("Start Training")
        self.train()
        print("End Training")
        self.predict()
        if self.save_model_bool:
            self.save_model()

    def run_grid_search(self):
        """
        Run a grid search on AdaBoostClassifier with decision tree
        """
        print("Start Grid Training")
        model =  AdaBoostClassifier(n_estimators= self.n_estim,
                                    base_estimator= DecisionTreeClassifier(),
                                    learning_rate= self.lr )
        parameters = {'n_estimators': (100, 200, 400, 600),
                      'base_estimator__max_depth': (1, 2, 3),
                      'learning_rate': (0.05, 0.1, 0.5, 0.75)}
        self.grid_search = GridSearchCV(model, parameters, scoring = 'accuracy')
        self.grid_search.fit(self.data["input_train"], self.data["output_train"])
        self.display_grid_search_result()

    def display_grid_search_result(self)-> None:
        """
        Print some result from grid search and saves them to a text file in the Result directory
        """
        with open(os.path.join(self.result_path, 'grid_search_result.txt'), 'w') as f:
            print(self.grid_search.best_params_)
            f.write("Best parameters: %s\n" % str(self.grid_search.best_params_))
            print("Grid scores on development set:")
            f.write("Grid scores on development set:\n")
            print()
            means = self.grid_search.cv_results_['mean_test_score']
            stds = self.grid_search.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, self.grid_search.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
                f.write("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))
            print()
            f.write(" \n")
            print("Detailed classification report:")
            f.write("Detailed classification report:\n")
            print()
            print("The model is trained on the full development set.")
            f.write("The model is trained on the full development set.\n")
            print("The scores are computed on the full evaluation set.")
            f.write("The scores are computed on the full evaluation set.\n")
            print()
            y_pred = self.grid_search.predict(self.data["input_test"])
            print(metrics.classification_report(self.data["output_test"], y_pred))
            f.write(metrics.classification_report(self.data["output_test"], y_pred))

    def save_model(self)-> None:
        """
        Save the model in the Result folder using joblib
        """
        dump(self.classifier, os.path.join(self.result_path, 'saved_model.joblib'))

    def load_model(self, model_path)->None:
        """
        Load a model from a model_path (must end by .joblib
        """
        self.classifier = load('filename.joblib')
