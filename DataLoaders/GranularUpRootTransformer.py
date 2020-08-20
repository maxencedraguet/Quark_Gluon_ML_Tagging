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
import random
import numpy as np
import pandas as pd
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
        self.save_csv_bool = config.get(["GranularUpRootTransformer", "to_CSV"])
        self.save_hdf_bool = config.get(["GranularUpRootTransformer", "to_HDF5"])
        print("Saving to H5? ", self.save_hdf_bool)
        self.clean_jets_bool = config.get(["GranularUpRootTransformer", "clean_jets"])
        
        self.do_JUNIPR_transform_bool = config.get(["GranularUpRootTransformer", "JUNIPR_transform"])
        self.save_JUNIPR_transform_bool = config.get(["GranularUpRootTransformer", "save_JUNIPR_transform"])
        self.JUNIPR_cluster_algo= config.get(["GranularUpRootTransformer", "JUNIPR_cluster_algo"])
        self.JUNIPR_cluster_radius= config.get(["GranularUpRootTransformer", "JUNIPR_cluster_radius"])
        
        self.cut_train_test= config.get(["GranularUpRootTransformer", "cut_train_test"])
        self.test_size = config.get(["GranularUpRootTransformer", "test_size"])
        self.add_cut = config.get(["GranularUpRootTransformer", "add_cut"])
        self.special_cut_junipr_bool = config.get(["GranularUpRootTransformer", "special_cut_junipr", "special_cut_bool"])
        self.special_cut_junipr_param = config.get(["GranularUpRootTransformer", "special_cut_junipr", "param"])
        self.purify_cut_bool = config.get(["GranularUpRootTransformer", "purify_cut"])
        self.pure_label_required  = config.get(["GranularUpRootTransformer", "purify_cut_val"])
        
        
        self.energy_histogram_limitor_path = config.get(["GranularUpRootTransformer", "energy_histogram_limitor_path"])
        self.energy_histogram_limitor_bool = config.get(["GranularUpRootTransformer", "energy_histogram_limitor"])

        self.save_path_junipr = os.path.join(self.path, 'junipr/')
        self.save_path_csv = os.path.join(self.path, 'CSV/')
        self.save_path_hf = os.path.join(self.path, 'HF/')
        self.diagnostic_path = os.path.join(self.path, 'Diagnostic/')
        
        self.seed = config.get(["seed"])
        random.seed(self.seed)
        
        self.diagnostic_bool = config.get(["diagnostic"])
    
    def get_inputs_list(self)->None:
        """
        Given a file with a list of target root file (only those uncommented).

        Input: file containing directories paths
        Output: a list of children directories from (uncommented) fed directories
        """
        self.inputs_list = []
        with open(self.data_file, 'r') as input_file:
            for dir_path in input_file.readlines():
                dir_path = dir_path.rstrip('\n')
                if not dir_path.startswith("#"):
                    self.inputs_list += [dir_path]
                    print (dir_path)

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
            - Removes jets with a pT inferior to 20 GeV.
        Removes all constituent linked to jets matching these conditions.
        
        Note: removeing jets that have no quark/gluon truth info (isTruthQuark -1) is not done by fear of biasing
              datasets and also because data could not be treated this way.
              
        Returns the two dataframes filtered.
        """
        c_pdf = constituent_pdf.copy(deep = True)
        j_pdf = jet_pdf.copy(deep = True)

        # Small check: no constituent should have a negative energy. If this happens, drop these constituents
        if (any(c_pdf['constituentE']) < 0):
            print("There are some constituents with negative energy!")
            bad_constituent_indices = c_pdf[(c_pdf['constituentE'] <= 0.0)].index
            print("The bad constituents: \n", c_pdf[bad_constituent_indices])
            table_of_bad_events = c_pdf[['entry', 'constituentJet']].iloc[bad_constituent_indices, :]
            
            # We need to know make table_of_bad_events unique (each unique row represents an event failing
            # and left-merge it to c_pdf. We will then remove from this merge the value that are matched between the two,
            # keeping only entries of c_pdf that had no matched in table_of_bad_events: meaning the good events
            c_pdf_all = c_pdf.merge(table_of_bad_events.drop_duplicates(),
                                    on=['entry', 'constituentJet'],
                                    how='left', indicator=True)
            j_pdf_all = pd.merge(j_pdf, table_of_bad_events.drop_duplicates(),
                                    left_on=['entry', 'subentry'],
                                    right_on=['entry', 'constituentJet'],
                                    how='left', indicator=True)
            # the "indicator=True" creates a column "_merge" with entries 'left_only' only for rows of c_pdf not contained in table_of_bad_events: our good events
            good_events_const_index = (c_pdf_all['_merge'] == 'left_only')
            c_pdf_all = c_pdf_all[good_events_const_index]
            c_pdf_all.drop(['_merge'], axis = 1, inplace=True) #drop the merge column
            
            good_events_jet_index = (j_pdf_all['_merge'] == 'left_only')
            j_pdf_all = j_pdf_all[good_events_jet_index]
            j_pdf_all.drop(['_merge', 'constituentJet'], axis = 1, inplace=True) #drop the merge and constituentJet (coming from constituent table) columns
            
            # return the modified database
            c_pdf = c_pdf_all
            j_pdf = j_pdf_all
        
        print("Initial shape jet ", j_pdf.shape)
        
        if self.clean_jets_bool:
            if self.purify_cut_bool:
                print("Restricting to a pure sample of label {}".format(self.pure_label_required))
                drop_bad_jet_indices = j_pdf[(j_pdf['isTruthQuark'] != self.pure_label_required)  |
                                             (j_pdf['jetPt'] < 20)       |
                                             (j_pdf['isNotPVJet'] == 1)  |      # This means your jet is not at the primary vertex (it's on another vertex)
                                             (j_pdf['jetNumberConstituent'] <= 4)].index
            else:
                drop_bad_jet_indices = j_pdf[(j_pdf['jetPt'] < 20)       |
                                             (j_pdf['isNotPVJet'] == 1)  |      # This means your jet is not at the primary vertex (it's on another vertex)
                                             (j_pdf['jetNumberConstituent'] <= 4)].index
            j_pdf.drop(drop_bad_jet_indices, inplace=True)
            if self.add_cut == "isolate_q_truth":
                drop_bad_jet_indices = j_pdf[(j_pdf['isTruthQuark'] !=1)].index
                j_pdf.drop(drop_bad_jet_indices, inplace=True)
            elif self.add_cut == "isolate_g_truth":
                drop_bad_jet_indices = j_pdf[(j_pdf['isTruthQuark'] !=0)].index
                j_pdf.drop(drop_bad_jet_indices, inplace=True)
            elif self.add_cut == "pt_above_peak":
                drop_bad_jet_indices = j_pdf[(j_pdf['jetPt'] < 300)].index
                j_pdf.drop(drop_bad_jet_indices, inplace=True)
            elif self.add_cut == "pt_below_peak":
                drop_bad_jet_indices = j_pdf[(j_pdf['jetPt'] > 300)].index
                j_pdf.drop(drop_bad_jet_indices, inplace=True)

        j_pdf_small = j_pdf[['entry', 'subentry', 'isTruthQuark']]
        print("Final shape jet ", j_pdf.shape)
        print("Initial shape constituent ", c_pdf.shape)
        c_pdf = pd.merge(c_pdf, j_pdf_small, how='inner', left_on=['entry', 'constituentJet'], right_on=['entry', 'subentry'])
        print("Final shape constituent ", c_pdf.shape)

        return c_pdf, j_pdf

    def energy_cleaning(self, constituent_pdf, jet_pdf):
        """
        This restricts the data to a given number of jet per energy bin.
        This number per bin is loaded from the self.energy_histogram_limitor_path.
        """
        print("Doing an extra cutting step to restrict the energy distribution to that of {}".format(self.energy_histogram_limitor_path))
        
        jet_pdf['jetE_binned'] = np.floor(jet_pdf['jetE'])
        limiting_array = np.loadtxt(self.energy_histogram_limitor_path, delimiter=',', dtype='int')[0, :]
        #resulting_j_pdf = pd.DataFrame(columns = jet_pdf.columns)
        resulting_j_pdf = pd.DataFrame({i[0]: pd.Series(dtype=i[1]) for i in jet_pdf.dtypes.iteritems()}, columns=jet_pdf.dtypes.index)


        for bin_idx, bin_value in enumerate(limiting_array):
            if bin_value != 0:
                if bin_idx == 4000:
                    sub_data_to_add = jet_pdf[jet_pdf['jetE_binned'] >= float(bin_idx)][:bin_value]
                else:
                    sub_data_to_add = jet_pdf[jet_pdf['jetE_binned'] == float(bin_idx)][:bin_value]
                if len(sub_data_to_add) == 0:
                    print("Empty at bin {} and expecting {}".format(bin_idx, bin_value))
                    continue
                resulting_j_pdf = resulting_j_pdf.append(sub_data_to_add, ignore_index=True)

        resulting_j_pdf.drop(['jetE_binned'], axis = 1, inplace=True) #drop the discretised column
        j_pdf_small = resulting_j_pdf[['entry', 'subentry']]

        print("Shape before energy cut of jet ", jet_pdf.shape)
        print("Shape after energy cut of jet ", resulting_j_pdf.shape)
        print("Shape before energy cut of constituent ", constituent_pdf.shape)
        c_pdf = pd.merge(constituent_pdf, j_pdf_small, how='inner', left_on=['entry', 'constituentJet'], right_on=['entry', 'subentry'])
        print("Shape after energy cut of ", c_pdf.shape)
    
        return c_pdf, resulting_j_pdf
    
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
            file_name = file_name.split(".")[:5]
            file_name = "_".join(file_name)
            print("Processing : ", file_name)
            print("the whole file is ", file)

            # List of variables
            constituent_vars = Specific_Set4_Parameters.qg_constituent_vars
            jet_vars = Specific_Set4_Parameters.qg_jet_vars
            
            # Get the dataframes from root file
            constituent_pdf = self.make_df(file, "analysis", constituent_vars)
            jet_pdf = self.make_df(file, "analysis", jet_vars)
            
            constituent_pdf.reset_index(inplace=True)
            jet_pdf.reset_index(inplace=True)
            
            # Translate MeV distributions into GeV ones
            for var in Specific_Set4_Parameters.vars_convert_MeV_to_GeV_constituent:
                constituent_pdf[var] =  constituent_pdf[var].div(1000)
            for var in Specific_Set4_Parameters.vars_convert_MeV_to_GeV_jet:
                jet_pdf[var] =  jet_pdf[var].div(1000)
            
            if self.diagnostic_bool:
                # These happen before cleaning
                self.diagnostic_plots(constituent_pdf, file_name, ['constituentE'], additional_text = "_Before_cuts")
                self.diagnostic_plots(jet_pdf, file_name, ['isNotPVJet', 'isTruthQuark', 'jetSumTrkPt500', 'jetNumTrkPt500'], additional_text = "_Before_cuts")
            
            constituent_pdf, jet_pdf = self.event_cleaning(constituent_pdf, jet_pdf)
            
            if self.energy_histogram_limitor_bool:
                
                if self.diagnostic_bool:
                    # These happen before the final cleaning
                    self.diagnostic_plots(constituent_pdf, file_name, ['constituentE'], additional_text = "_Before_Energy_cuts")
                    self.diagnostic_plots(jet_pdf, file_name, ['isNotPVJet', 'isTruthQuark', 'jetSumTrkPt500', 'jetNumTrkPt500', 'jetE', 'jetPt', 'jetEta','jetPhi','jetMass'], additional_text = "_Before_Energy_cuts")
                constituent_pdf, jet_pdf = self.energy_cleaning(constituent_pdf, jet_pdf)
                print("Is anything null in jet pdf after energy clean ? ",jet_pdf.isnull().values.any())
    
            if self.save_tables_bool:
                if self.save_csv_bool:
                    #self.save_to_csv(constituent_pdf, file_name+"_constituents")
                    self.save_to_csv(jet_pdf, file_name+"_jets")
                
                if self.save_hdf_bool:
                    print("Saving to H5")
                    #self.save_to_h5(constituent_pdf, file_name+"_constituents")
                    self.save_to_h5(jet_pdf, file_name+"_jets")
        
            if self.diagnostic_bool:
                self.diagnostic_plots(constituent_pdf, file_name, constituent_vars)
                print("Available jet info: ", jet_pdf.columns)
                self.diagnostic_plots(jet_pdf, file_name, jet_vars)
                # Some constituent to jet comparison
                self.compare_dataset_info(jet_pdf, constituent_pdf, file_name)
            
            if self.do_JUNIPR_transform_bool:
                start = time.process_time()
                dictionnary_result, dictionnary_exception  = perform_clustering(self.JUNIPR_cluster_algo, self.JUNIPR_cluster_radius, jet_pdf, constituent_pdf)
                print("Time for jet clustering to JUNIPR {}".format(time.process_time() - start))
                if self.save_JUNIPR_transform_bool:
                    self.save_junipr_data_to_json(dictionnary_exception, file_name+"_EXCEPTIONS")
                    if self.cut_train_test:
                        list_of_jets = dictionnary_result["JuniprJets"]
                        random.shuffle(list_of_jets)
                        size_of_data_test = int(len(list_of_jets) * self.test_size)
                        #print("size_of_data_test = ", size_of_data_test)
                        #print("Type of dictionnary_result ", type(dictionnary_result))
                        list_train = list_of_jets[size_of_data_test:]
                        list_test  = list_of_jets[:size_of_data_test]
                        dictionnary_result_train = {"JuniprJets": list_train}
                        dictionnary_result_test  = {"JuniprJets": list_test}
                        
                        self.save_junipr_data_to_json(dictionnary_result_train, file_name+"_train")
                        self.save_junipr_data_to_json(dictionnary_result_test, file_name+"_test")
                    else:
                        self.save_junipr_data_to_json(dictionnary_result, file_name)
            
        warnings.filters = original_warnings

    def save_to_csv(self, pdf, file_name)->None:
        """
        Save a panda df to csv in a CSV folder in save_path_csv.
        """
        os.makedirs(self.save_path_csv, exist_ok=True)
        pdf.to_csv(os.path.join(self.save_path_csv, file_name + '.csv'))
    
    def save_to_h5(self, pdf, file_name)->None:
        """
        Save a panda df to hdf5 in a HF folder in save_path_hf. Several files would be copied in the same HF using the name of the file as key.
        """
        os.makedirs(self.save_path_hf, exist_ok=True)
        pdf.to_hdf(os.path.join(self.save_path_hf, file_name + '.h5'), key = file_name)

    def save_junipr_data_to_json(self, dictionnary, file_name, bool_no_indent = True):
        """
        Saves a junipr-ready data dicitonnay to a json file located self.save_path_junipr/ + file_name.json
        """
        os.makedirs(self.save_path_junipr, exist_ok=True)
        #file_name = "example_JUNIPR_data_CA"
            #with open( os.path.join(self.save_path_junipr, file_name+ ".json"), "w") as write_file:
        if bool_no_indent:
            with open( os.path.join(self.save_path_junipr, file_name + '.json'), "w") as write_file:
                json.dump(dictionnary, write_file) #, indent=4)
        else:
            with open( os.path.join(self.save_path_junipr, file_name + '.json'), "w") as write_file:
                json.dump(dictionnary, write_file, indent=4)
    

    def diagnostic_plots(self, df, file_name, vars, additional_text = ""):
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
            #if plot_name == 'jetNumTrkPt500':
            #print(df[plot_name])
            fig = plt.figure(p)
            ax = fig.add_subplot(1, 1, 1)
            df.hist(column=plot_name,
                    bins = Specific_Set4_Parameters.plot_xbins[plot_name],
                    range = Specific_Set4_Parameters.plot_xranges[plot_name],
                    ax = ax)
            ax.text(0.7, 0.7, additional_text.replace("_", " "), fontsize=10,  horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
            ax.set_title(plot_name)
            ax.set_xlabel(Specific_Set4_Parameters.plot_xlabels[plot_name])
            ax.set_ylabel('Events')
            fig.savefig(os.path.join(local_path, plot_name + additional_text + '.png'), dpi=300, format='png', bbox_inches='tight')
            if plot_name in Specific_Set4_Parameters.log_hist:
                ax.set_yscale('log')
                fig.savefig(os.path.join(local_path, plot_name + additional_text + '_log.png'), dpi=300, format='png', bbox_inches='tight')
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
