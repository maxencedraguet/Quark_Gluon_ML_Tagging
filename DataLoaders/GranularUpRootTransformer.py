#############################################################################
#
# GranularUpRootTransformer.py
#
# A data loader for a given directory of Root files. Loads everything, based on a selection of variables.
#
# Author -- Maxence Draguet (26/06/2020)
#
# This loader loads root files through a text file indicating their (mother) directory
#             and converts them into the expected format using UpRoot.
#
# A list of variables of interest should be specified in Utils/Specific_Set4_Parameters.
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Notes regarding the datastructre:
# The organisation should strictly follow this:
# Directory/.../sub-Directory/RootFilesDirectory/X.root
# ! Any root files at the same level as a directory are discarded:
#        -> root files must be leaves and directories nodes.
# This is indeed the case for 20200506_qgTaggingSystematics/mc16a.
#
# Constituent of jet variables accessible:
#
# ['constituentE', 'constituentPt', 'constituentEta', 'constituentPhi', 'constituentMass', 'constituentDeltaRtoJet',
#  'constituentJet', 'constituentRunNumber', 'constituentEventNumber']
#
# Jet variables accessible:
#
# ['jetPt', 'jetEta', 'jetPhi', 'jetMass', 'jetEnergy', 'jetEMFrac', 'jetHECFrac', 'jetChFrac',
# 'jetNumTrkPt500', 'jetNumTrkPt1000', 'jetTrackWidthPt500', 'jetTrackWidthPt1000', 'jetSumTrkPt500',
# 'jetSumTrkPt1000', 'partonIDs', 'BDTScore', 'isTruthQuark', 'isBDTQuark', 'isnTrkQuark']
#
#############################################################################
import os
import sys
import warnings
import tables
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 5)
import uproot as upr
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import fnmatch
import pprint


from Utils import Specific_Set4_Parameters

class GranularUpRootTransformer(ABC):
    def __init__(self, config: Dict):
        self.extract_parameters(config)
        self.get_inputs_list()
        self.run_uproot()

    def extract_parameters(self, config: Dict)->None:
        # In this case it should be the path to a text file, with all directories to load.
        self.data_file = config.get(["absolute_data_path"]) # does not matter right now.
        self.path = config.get(["GranularUpRootTransformer", "save_path"])
        self.save_csv_bool = config.get(["GranularUpRootTransformer", "to_CSV"])
        self.save_hdf_bool = config.get(["GranularUpRootTransformer", "to_HDF5"])
        self.save_path_csv = os.path.join(self.path, 'CSV/')
        self.save_path_hf = os.path.join(self.path, 'HF/')
        self.diagnostic_path = os.path.join(self.path, 'Diagnostic/')
        
        self.seed = config.get(["seed"])
        self.diagnostic_bool = config.get(["diagnostic"])

    def get_inputs_list(self)->None:
        """
        NOTE: FOR NOW THIS ONLY READS ONE FILE
        Given a file with a list directories (only those uncommented), extract a list of all lowest children directories.
        Appropriate for loading in bulk root files contained in these children directories (note that any root file at the
        same level as a directory is discarded: only the leaves are loaded)
        
        Input: file containing directories paths
        Output: a list of children directories from (uncommented) fed directories
        """
        self.inputs_list = ["/data/atlas/atlasdata3/mdraguet/Set4/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_JETM6.e6337_e5984_s3126_r10201_r10210_p4128.root"]
    
    def make_df(self, input_file, tree_name, branches):
        """
        Turn the specified branches of the input tree from the input file
        into a single panda dataframe (appening rows at the end).
        """
        #print('Getting Tree:', tree_name)
        #print('With branches:', branches)
        dataframes = pd.DataFrame()
        for array in upr.iterate(input_file, tree_name, branches, outputtype=pd.DataFrame,flatten=True):
            dataframes = dataframes.append(array)
        return dataframes

    def event_cleaning(self, pdf):
        """
        Apply some cuts on the data.
        """
        pass

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

            # List of variables
            constituent_vars = Specific_Set4_Parameters.qg_constituent_vars
            jet_vars = Specific_Set4_Parameters.qg_jet_vars
            
            # Get the dataframes from root file
            constituent_pdf = self.make_df(file, "analysis", constituent_vars)
            jet_pdf = self.make_df(file, "analysis", jet_vars)
            
            constituent_pdf.reset_index(inplace=True)
            jet_pdf.reset_index(inplace=True)
            
            # Necessary if the constituentJet column has a counter starting at 1.
            constituent_pdf['constituentJet'] = constituent_pdf['constituentJet'] - 1
            
            # Some constituent to jet comparison
            self.compare_dataset_info(jet_pdf, constituent_pdf, file_name)
        
            if self.save_csv_bool:
                self.save_to_csv(constituent_pdf, file_name)
            if self.save_hdf_bool:
                self.save_to_h5(constituent_pdf, super_file_name, file_name)
            if self.diagnostic_bool:
                self.diagnostic_plots(constituent_pdf, file_name, constituent_vars)
                self.diagnostic_plots(jet_pdf, file_name, jet_vars)
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

    def diagnostic_plots(self, df, file_name, vars):
        """
        Performs some diagnostic plots and store them in self.diagnostic_path.
        """
        os.makedirs(self.diagnostic_path, exist_ok=True)
        local_path = os.path.join(self.diagnostic_path, file_name + '/')
        os.makedirs(local_path, exist_ok=True)
        for p, plot_name in enumerate(vars):
            if plot_name in Specific_Set4_Parameters.skip_hist:
                continue
            print(plot_name)
            fig = plt.figure(p)
            ax = fig.add_subplot(1, 1, 1)
            df.hist(column=plot_name,
                    bins = Specific_Set4_Parameters.plot_xbins[plot_name],
                    range = Specific_Set4_Parameters.plot_xranges[plot_name],
                    ax = ax)
            ax.set_title(plot_name)
            ax.set_xlabel(Specific_Set4_Parameters.plot_xlabels[plot_name])
            ax.set_ylabel('Events')
            fig.savefig(os.path.join(local_path, plot_name + '.png'), dpi=300, format='png', bbox_inches='tight')
            if plot_name in Specific_Set4_Parameters.log_hist:
                ax.set_yscale('log')
                fig.savefig(os.path.join(local_path, plot_name + '_log.png'), dpi=300, format='png', bbox_inches='tight')
            plt.close()

    def compare_dataset_info(self, jet_pdf, constituent_pdf, file_name):
        """
        Compares Energy and Momentum (px and py) of jet and constituents
        """
        os.makedirs(self.diagnostic_path, exist_ok=True)
        local_path = os.path.join(self.diagnostic_path, file_name + '/')
        os.makedirs(local_path, exist_ok=True)
        
        counter_df = constituent_pdf.copy(deep = True)
        counter_df = counter_df[['entry', 'constituentJet', 'constituentPt']].groupby(['entry', 'constituentJet'], sort=False)["constituentPt"].count().reset_index(name ='count')
        
        c_pdf = constituent_pdf.copy(deep = True)
        j_pdf = jet_pdf.copy(deep = True)
        #arrays = [ ]
        diff = pd.DataFrame()
        
        c_pdf['cPx'] = c_pdf['constituentPt'] * np.cos(c_pdf['constituentPhi'])
        c_pdf['cPy'] = c_pdf['constituentPt'] * np.sin(c_pdf['constituentPhi'])
        j_pdf['jPx'] = j_pdf['jetPt'] * np.cos(j_pdf['jetPhi'])
        j_pdf['jPy'] = j_pdf['jetPt'] * np.sin(j_pdf['jetPhi'])

        c_pdf = c_pdf[['entry', 'constituentJet', 'constituentE', 'cPx', 'cPy']].groupby(['entry', 'constituentJet']).sum()
        c_pdf.reset_index(inplace=True)
        
        diff = pd.merge(j_pdf, c_pdf, how='left', left_on=['entry', 'subentry'], right_on=['entry', 'constituentJet'])
        diff["differenceEnergy"] = diff['jetE'] - diff['constituentE']
        diff["differencePx"] = diff['jPx'] - diff['cPx']
        diff["differencePy"] = diff['jPy'] - diff['cPy']
        diff["CounterElem"] = counter_df ['count']

        for p, plot_name in enumerate(['differenceEnergy', 'differencePx', 'differencePy']):
            print(plot_name)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            diff.hist(column = plot_name,
                      bins = Specific_Set4_Parameters.plot_xbins[plot_name],
                      range = Specific_Set4_Parameters.plot_xranges[plot_name],
                      ax = ax)
            ax.set_title(plot_name)
            ax.set_xlabel(Specific_Set4_Parameters.plot_xlabels[plot_name])
            ax.set_ylabel('Count')
            fig.savefig(os.path.join(local_path, plot_name + '.png'), dpi=300, format='png', bbox_inches='tight')
            if plot_name in Specific_Set4_Parameters.log_hist:
                ax.set_yscale('log')
                fig.savefig(os.path.join(local_path, plot_name + '_log.png'), dpi=300, format='png', bbox_inches='tight')
            plt.close()

