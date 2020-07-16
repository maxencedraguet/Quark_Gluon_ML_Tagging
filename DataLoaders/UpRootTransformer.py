#############################################################################
#
# UpRootLoader.py
#
# A data loader for a given directory of Root files. Loads everything, based on a selection of variables.
#
# Code inspired by: Aaron O'Neill, https://github.com/aponeill89/MLForQgTaggingAndRPVSUSY/.
#
# Author -- Maxence Draguet (21/05/2020)
#
# This loader loads root files through a text file indicating their (mother) directory
#             and converts them into the expected format using UpRoot.
#
# A list of variables of interest should be specified in Utils/Specific_Set2_Parameters.
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Notes regarding the datastructre:
# The organisation should strictly follow this:
# Directory/.../sub-Directory/RootFilesDirectory/X.root
# ! Any root files at the same level as a directory are discarded:
#        -> root files must be leaves and directories nodes.
# This is indeed the case for 20200506_qgTaggingSystematics/mc16a.
#
#############################################################################

import os
import sys
import warnings
import tables
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd         #pd.set_option('display.max_rows', None)
import uproot as upr
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import fnmatch
import pprint


from Utils import Specific_Set2_Parameters

class UpRootTransformer(ABC):
    def __init__(self, config: Dict):
        self.extract_parameters(config)
        self.get_inputs_list()
        self.run_uproot()
    
    def extract_parameters(self, config: Dict)->None:
        # In this case it should be the path to a text file, with all directories to load.
        self.data_file = config.get(["absolute_data_path"])
        self.path = config.get(["UpRootTransformer", "save_path"])
        self.save_csv_bool = config.get(["UpRootTransformer", "to_CSV"])
        self.save_hdf_bool = config.get(["UpRootTransformer", "to_HDF5"])
        self.save_path_csv = os.path.join(self.path, 'CSV/')
        self.save_path_hf = os.path.join(self.path, 'HF/')
        self.diagnostic_path = os.path.join(self.path, 'Diagnostic/')
        
        self.seed = config.get(["seed"])
        self.diagnostic_bool = config.get(["diagnostic"])
    
    def get_inputs_list(self)->None:
        """
        Given a file with a list directories (only those uncommented), extract a list of all lowest children directories.
        Appropriate for loading in bulk root files contained in these children directories (note that any root file at the
        same level as a directory is discarded: only the leaves are loaded)
        
        Input: file containing directories paths
        Output: a list of children directories from (uncommented) fed directories
        """
        self.inputs_list = []
        self.directory_list = []
        
        # Read the text file to get the directories from which to load all root files
        with open(self.data_file, 'r') as input_file:
            print('Finding your directories...')
            for dir_path in input_file.readlines():
                dir_path = dir_path.rstrip('\n')
                if not dir_path.startswith("#"):
                    self.directory_list += [dir_path]
                print(dir_path)
        # Get all the lowest level directories
        for dir_path in self.directory_list:
            for path,sub_directories,_ in os.walk(dir_path):
                if not sub_directories:
                        if any(file.endswith('.root') for file in os.listdir(path)):
                            self.inputs_list.append(path)
                            #print(path)
            
    def make_df(self, input_file, tree_name, branches):
        """
        Turn the specified branches of the input tree from the input file
        into a single panda dataframe (appening rows at the end).
        """
        #print('Getting Tree:', tree_name)
        #print('With branches:', branches)
        
        dataframes = pd.DataFrame()
        for array in tqdm(upr.iterate(input_file + "/*.root", tree_name, branches, outputtype=pd.DataFrame, flatten=True),total=len(fnmatch.filter(os.listdir(input_file), '*.root'))):
            dataframes = dataframes.append(array)
        
        return dataframes
        
    def event_cleaning(self, pdf):
        """
        Apply some cuts on the data.
        """
        
        # Start with data quality cuts (the most demanding filter)
        drop_indices_quality = pdf[(pdf['jetTrackWidthPt500'] < 0.0)  |
                                   (pdf['jetTrackWidthPt1000'] < 0.0) |
                                   (pdf['isTruthQuark'] < 0)].index
        pdf.drop(drop_indices_quality , inplace=True)
                                       
        drop_indices = pdf[(pdf['numPrimaryVertices'] == 0) |
                           (pdf['numPrimaryVertices'] == 0) |
                           (pdf['hasBadMuon'] == 1)         |
                           (pdf['hasCosmic'] == 1)          |
                           (pdf['PVnumTrk'] < 2)].index
        pdf.drop(drop_indices , inplace=True)
        # pdf.drop(pdf[pdf['GenFiltHT'] < 600].index, inplace=True)
        
        # Only keep variables required.
        pdf.drop(pdf.columns.difference(Specific_Set2_Parameters.qg_tagging_vars), axis = 1, inplace=True)
        
        # More analysis oreintated cuts, 'baseline jets'.
        drop_indices_analysis = pdf[(pdf['jetPt'] < 20)          |
                                    (pdf['jetEta'].abs() > 2.5)  |
                                    (pdf['BDTScore'] == -666.0)].index       # Get rid of nasty values in the BDT and truth information
        pdf.drop(drop_indices_analysis , inplace=True)

        # Turns variables given in MeV into GeV.
        pdf['jetSumTrkPt500']  = pdf['jetSumTrkPt500'].div(1000)
        pdf['jetSumTrkPt1000'] = pdf['jetSumTrkPt1000'].div(1000)
        pdf['jetEnergy']       = pdf['jetEnergy'].div(1000)

        return pdf
    
    def run_uproot(self):
        """
        Execute the loading, filtering and (potentially) saving and displaying.
        """
        # The naming system for the hdf5 storing system generate a warning given the use of ".".
        # This is no trouble for loading so this is discarded here.
        original_warnings = list(warnings.filters)
        warnings.simplefilter('ignore', tables.NaturalNameWarning)
        for file in self.inputs_list:
            # Get the filename: the lowest directory (where the root files are stored).
            file_name = file.split("/")[-1]
            super_file_name = file.split("/")[-2]
            print("Processing : ", file_name)
            
            # List of variables for cleaning
            clean_vars_list = Specific_Set2_Parameters.nominal_cleaning_vars + Specific_Set2_Parameters.qg_tagging_vars

            # Get the dataframes from root file
            common_tree_pdf = self.make_df(file, "commonValues", Specific_Set2_Parameters.common_cleaning_vars)

            common_tree_pdf.index.names = ['entry']
            nominal_tree_pdf = self.make_df(file, "Nominal", clean_vars_list)
            combined_pdf = nominal_tree_pdf.join(common_tree_pdf)
            
            # Clean events
            combined_pdf = self.event_cleaning(combined_pdf)
            
            if self.save_csv_bool:
                self.save_to_csv(combined_pdf, file_name)
            if self.save_hdf_bool:
                self.save_to_h5(combined_pdf, super_file_name, file_name)
            if self.diagnostic_bool:
                self.diagnostic_plots(combined_pdf, file_name)
        warnings.filters = original_warnings

    def save_to_csv(self, pdf, file_name)->None:
        """
        Save a panda df to csv in a CSV folder in save_path_csv.
        """
        os.makedirs(self.save_path_csv, exist_ok=True)
        pdf.to_csv(os.path.join(self.save_path_csv, file_name + '_all.csv'))
    
    def save_to_h5(self, pdf, super_file_name, file_name)->None:
        """
        Save a panda df to hdf5 in a HF folder in save_path_hf. Several files would be copied in the same HF using the name of the file as key.
        """
        os.makedirs(self.save_path_hf, exist_ok=True)
        pdf.to_hdf(os.path.join(self.save_path_hf, super_file_name+ 'processed.h5'), key = file_name)

    def diagnostic_plots(self, df, file_name):
        """
        Performs some diagnostic plots and store them in self.diagnostic_path.
        """
        os.makedirs(self.diagnostic_path, exist_ok=True)
        local_path = os.path.join(self.diagnostic_path, file_name + '/')
        os.makedirs(local_path, exist_ok=True)
        for p, plot_name in enumerate(Specific_Set2_Parameters.qg_tagging_vars):
            plt.figure(p)
            df.hist(column=plot_name)
            plt.title(plot_name)
            plt.xlabel(plot_name)
            plt.ylabel('Events')
            plt.savefig(os.path.join(local_path, plot_name + '.png'))
            plt.close()
