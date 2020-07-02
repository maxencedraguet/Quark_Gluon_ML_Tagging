#############################################################################
#
# GranularUpRootTransformer.py
#
# A data loader for a given directory of Root files. Loads everything, based on a selection of variables.
#
# Author -- Maxence Draguet (26/06/2020)
#
# This loader loads root files through a text file indicating their (mother) directory
#             and converts them into the expected format using UpRoot and pyjet.
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
from typing import Dict, List

import math
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_colwidth', -1)
import uproot as upr
from pyjet import cluster
from pyjet.utils import ptepm2ep
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import fnmatch
import pprint
import json
import time

from Utils import Specific_Set4_Parameters
from .JuniprProcessing import *

class GranularUpRootTransformer(ABC):
    def __init__(self, config: Dict):
        very_start = time.process_time()
        self.extract_parameters(config)
        self.get_inputs_list()
        self.run_uproot()
        print("Time for whole Granular {}".format(time.process_time() - very_start))

    def extract_parameters(self, config: Dict)->None:
        # In this case it should be the path to a text file, with all directories to load.
        self.data_file = config.get(["absolute_data_path"]) # does not matter right now.
        
        self.path = config.get(["GranularUpRootTransformer", "save_path"])
        self.save_tables_bool = config.get(["GranularUpRootTransformer", "save_tables"])
        self.do_JUNIPR_transform_bool = config.get(["GranularUpRootTransformer", "JUNIPR_transform"])
        self.save_JUNIPR_transform_bool = config.get(["GranularUpRootTransformer", "save_JUNIPR_transform"])
        self.save_csv_bool = config.get(["GranularUpRootTransformer", "to_CSV"])
        self.save_hdf_bool = config.get(["GranularUpRootTransformer", "to_HDF5"])
        
        self.clean_jets_bool = config.get(["GranularUpRootTransformer", "clean_jets"])
        add_to_path = ""
        if self.clean_jets_bool:
            add_to_path = "cut_"
        self.save_path_junipr = os.path.join(self.path, 'junipr/')
        self.save_path_csv = os.path.join(self.path, 'CSV/')
        self.save_path_hf = os.path.join(self.path, 'HF/')
        self.diagnostic_path = os.path.join(self.path, add_to_path+'Diagnostic/')
        
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

    def event_cleaning(self, constituent_pdf, jet_pdf):
        """
        Apply some cuts and cleaning on the data:
            - Removes jets with 3 or less constituents
            - Removes jets that have no quark/gluon truth info (isTruthQuark -1)
            - Removes jets with a pT inferior to 20 GeV.
        Removes all constituent linked to jets matching these conditions.
        
        Returns the two dataframes filtered.
        """
        c_pdf = constituent_pdf.copy(deep = True)
        j_pdf = jet_pdf.copy(deep = True)
        # Translate MeV distributions into GeV ones
        for var in Specific_Set4_Parameters.vars_convert_MeV_to_GeV_constituent:
            c_pdf[var] =  c_pdf[var].div(1000)
        for var in Specific_Set4_Parameters.vars_convert_MeV_to_GeV_jet:
            j_pdf[var] =  j_pdf[var].div(1000)
        
        print("Initial shape jet ", j_pdf.shape)
        if self.clean_jets_bool:
            drop_bad_jet_indices = j_pdf[(j_pdf['jetPt'] < 20)  |
                                         (j_pdf['isTruthQuark'] < 0) |
                                         (j_pdf['jetNumberConstituent'] <= 3)].index
            j_pdf.drop(drop_bad_jet_indices , inplace=True)
        j_pdf_small = j_pdf[['entry', 'subentry', 'isTruthQuark']]
        print("Final shape jet ", j_pdf.shape)
        print("Initial shape constituent ", c_pdf.shape)
        c_pdf = pd.merge(c_pdf, j_pdf_small, how='inner', left_on=['entry', 'constituentJet'], right_on=['entry', 'subentry'])
        print("Final shape constituent ", c_pdf.shape)
        return c_pdf, j_pdf
    
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
            
            constituent_pdf, jet_pdf = self.event_cleaning(constituent_pdf, jet_pdf)
            
            if self.save_tables_bool:
                if self.save_csv_bool:
                    self.save_to_csv(constituent_pdf, file_name)
                if self.save_hdf_bool:
                    self.save_to_h5(constituent_pdf, super_file_name, file_name)
            if self.diagnostic_bool:
                self.diagnostic_plots(constituent_pdf, file_name, constituent_vars)
                self.diagnostic_plots(jet_pdf, file_name, jet_vars)
                # Some constituent to jet comparison
                self.compare_dataset_info(jet_pdf, constituent_pdf, file_name)
            if self.do_JUNIPR_transform_bool:
                start = time.process_time()
                dictionnary_result = perform_antiKT(jet_pdf, constituent_pdf)
                print("Time for antiKT {}".format(time.process_time() - start))
                if self.save_JUNIPR_transform_bool:
                    self.save_junipr_data_to_json(dictionnary_result, file_name)
            
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

    def save_junipr_data_to_json(self, dictionnary, file_name):
        """
        Saves a junipr-ready data dicitonnay to a json file located self.save_path_junipr/ + file_name.json
        """
        os.makedirs(self.save_path_junipr, exist_ok=True)
            #with open( os.path.join(self.save_path_junipr, file_name+ ".json", "w")) as write_file:
        with open( file_name+ ".json", "w") as write_file:
            json.dump(dictionnary, write_file, indent=4)

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

        c_pdf = c_pdf[['entry', 'constituentJet', 'constituentE', 'cPx', 'cPy', 'constituentPx', 'constituentPy']].groupby(['entry', 'constituentJet']).sum()
        c_pdf.reset_index(inplace=True)
        
        diff = pd.merge(j_pdf, c_pdf, how='left', left_on=['entry', 'subentry'], right_on=['entry', 'constituentJet'])
        diff["differenceEnergy"] = diff['jetE'] - diff['constituentE']
        diff["differencePx_computed"] = diff['jPx'] - diff['cPx']
        diff["differencePy_computed"] = diff['jPy'] - diff['cPy']
        diff["differencePx"] = diff['jPx'] - diff['constituentPx']
        diff["differencePy"] = diff['jPy'] - diff['constituentPy']
        diff["CounterElem"] = counter_df ['count']

        for p, plot_name in enumerate(['CounterElem', 'differenceEnergy', 'differencePx', 'differencePy', 'differencePx_computed', 'differencePy_computed']):
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
