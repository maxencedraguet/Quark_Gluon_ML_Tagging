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
import matplotlib.pyplot as plt
from typing import Dict
from joblib import dump, load

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

from DataLoaders import DataLoader_Set1, DataLoader_Set2
from .BaseRunner import _BaseRunner
from Utils import write_ROC_info

class BDTRunner(_BaseRunner):
    def __init__(self, config: Dict) -> None:
        self.extract_parameters(config)
        self.classifier = AdaBoostClassifier(n_estimators= self.n_estim,
                                        base_estimator= self.base_estimator,
                                        learning_rate= self.lr )
        self.setup_dataset(config)
        
        if self.grid_search_bool:
            self.run_grid_search()
        else:
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
        
        self.grid_search_bool = config.get(["BDT_model", "grid_search"])
        self.n_estim = config.get(["BDT_model", "n_estim"])
        self.base_estimator = config.get(["BDT_model", "base_estimator"])
        self.max_depth = config.get(["BDT_model", "max_depth"])
        self.lr = config.get(["BDT_model", "lr"])
        self.extract_BDT_info = config.get(["NN_Model", "extract_BDT_info"])
        
        if self.base_estimator == "DecisionTreeClassifier" and self.max_depth:
            self.base_estimator = DecisionTreeClassifier(max_depth = self.max_depth)

    def checkpoint_df(self, step: int) -> None:
        raise NotImplementedError("Base class method")
    
    def train(self)->None:
        self.model = self.classifier.fit(self.data["input_train"], self.data["output_train"])
        self.data["training_predictions"] = self.model.predict(self.data["input_train"])
        self.training_accuracy = metrics.accuracy_score(self.data["output_train"], self.data["training_predictions"])
    
    def test(self)->None:
        self.data["test_predictions"] = self.model.predict(self.data["input_test"])
        self.data["test_predictions_proba"] = self.model.predict_proba(self.data["input_test"])
        self.accuracy = metrics.accuracy_score(self.data["output_test"], self.data["test_predictions"])
        self.precision = metrics.average_precision_score(self.data["output_test"], self.data["test_predictions"])
        #confusion_matrix = metrics.confusion_matrix(self.data["output_test"], self.data["test_predictions"])
        self.assess()
        with open(os.path.join(self.result_path, 'fit_result.txt'), 'w') as f:
            f.write("The training accuracy obtained is: %s\n" % str(self.training_accuracy))
            f.write("The test accuracy obtained is: %s\n" % str(self.accuracy))
            f.write("The test precision obtained is: %s\n" % str(self.precision))
        print("The training accuracy obtained is ", self.training_accuracy)
        print("The test accuracy obtained is ", self.accuracy)
        print("The test precision obtained is ", self.precision)
    
    def assess(self)->None:
        """
        Produce some metric of the result:
        - Confusion Matrices (normalised and absolute)
        - ROC curve (versus BDT furnished)
        """
        # Confusion Matrix
        np.set_printoptions(precision=2)
        plot_mat = metrics.plot_confusion_matrix(self.model, self.data["input_test"], self.data["output_test"],
                                                 cmap=plt.cm.Blues)
        plot_mat.ax_.set_title("Confusion Matrix BDT")
        plt.savefig(os.path.join(self.result_path, 'confusion_matrix.png'))
        plt.close()
        
        # Confusion Matrix Normalised
        plot_mat_n = metrics.plot_confusion_matrix(self.model, self.data["input_test"], self.data["output_test"],
                                                   cmap=plt.cm.Blues, normalize='true')
        plot_mat_n.ax_.set_title("Confusion Matrix Normalised BDT")
        plt.savefig(os.path.join(self.result_path, 'confusion_matrix_normalised.png'))
        plt.close()
        
        # ROC curve
        write_ROC_info(os.path.join(self.result_path, 'test_label_pred_proba_BDT.txt'),
                       self.data["output_test"], self.data["test_predictions_proba"][:,1])
        if self.extract_BDT_info:
            write_ROC_info(os.path.join(self.result_path, 'test_label_pred_proba_given_BDT.txt'),
                            self.data["output_test"], self.data["output_BDT_test"])
        false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(self.data["output_test"], self.data["test_predictions_proba"][:,1])
        if self.extract_BDT_info:
            false_pos_rateBDT, true_pos_rateBDT, thresholdsBDT = metrics.roc_curve(self.data["output_test"], self.data["output_BDT_test"])
        AUC_test = metrics.auc(false_pos_rate, true_pos_rate)
        print("AUC on test set for own BDT is {0}".format(AUC_test))
        if self.extract_BDT_info:
            AUC_test_BDT = metrics.auc(false_pos_rateBDT, true_pos_rateBDT)
            print("AUC on test set for given BDT {1}".format(AUC_test_BDT))

        # Plot the ROC curve
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(false_pos_rateBDT, true_pos_rateBDT, label='BDT (area = {:.4f})'.format(AUC_test_BDT))
        plt.plot(false_pos_rate, true_pos_rate, label='Own BDT (area = {:.4f})'.format(AUC_test))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC Curves')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.result_path, 'ROC_curve.png'))
        plt.close()
        

    def run(self)->None:
        print("Start Training")
        self.train()
        print("End Training")
        self.test()
        if self.save_model_bool:
            self.save_model()

    def run_grid_search(self)->None:
        """
        Run a grid search on AdaBoostClassifier with decision tree
        """
        print("Start Grid Training")
        model =  AdaBoostClassifier(n_estimators= self.n_estim,
                                    base_estimator= DecisionTreeClassifier(max_depth = self.max_depth),
                                    learning_rate= self.lr)

        parameters = {'n_estimators': (325, 350, 375, 400),
                      'learning_rate': (0.125, 0.15, 0.175, 0.2)}
        #'base_estimator__max_depth': (2, 3, 4): 3 was found to almost always be better

        print("Start training")
        self.grid_search = GridSearchCV(model, parameters, scoring = 'accuracy', n_jobs = -1)
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
        dump(self.model, os.path.join(self.result_path, 'saved_model.joblib'))

    def load_model(self, model_path)->None:
        """
        Load a model from a model_path (must end by .joblib
        """
        self.model = load('filename.joblib')
