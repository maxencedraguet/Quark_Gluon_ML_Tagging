#############################################################################
#
# GranularUpRootTransformerMP.py
#
# A multiprocessing equivalent ofGranularUpRootTransformer.py
# This one directly receives the path the file it should analyse
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
import fnmatch
import json
import time

from Utils import Specific_Set4_Parameters
from .JuniprProcessing import *


def make_df(input_file, tree_name, branches):
    """
    Turn the specified branches of the input tree from the input file
    into a single panda dataframe (appening rows at the end).
    """
    dataframes = pd.DataFrame()
    for array in upr.iterate(input_file, tree_name, branches, outputtype=pd.DataFrame,flatten=True):
        dataframes = dataframes.append(array)
    return dataframes

def event_cleaning(constituent_pdf, jet_pdf):
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
    
    #j_pdf = j_pdf.iloc[:2000, :]
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
    drop_bad_jet_indices = j_pdf[(j_pdf['jetPt'] < 20)       |
                                 (j_pdf['isNotPVJet'] == 1)  |      # This means your jet is not at the primary vertex (it's on another vertex)
                                 (j_pdf['jetNumberConstituent'] <= 4)].index
    j_pdf.drop(drop_bad_jet_indices, inplace=True)

    j_pdf_small = j_pdf[['entry', 'subentry', 'isTruthQuark']]
    print("Final shape jet ", j_pdf.shape)
    print("Initial shape constituent ", c_pdf.shape)
    c_pdf = pd.merge(c_pdf, j_pdf_small, how='inner', left_on=['entry', 'constituentJet'], right_on=['entry', 'subentry'])
    print("Final shape constituent ", c_pdf.shape)
    return c_pdf, j_pdf

def run_uproot(file, save_path):
    """
    Execute the loading, filtering and processing to JUNIPR as well as saving.
    """
    # The naming system for the hdf5 storing system generate a warning given the use of ".".
    # This is no trouble for loading so this is discarded here.
    original_warnings = list(warnings.filters)
    warnings.simplefilter('ignore', tables.NaturalNameWarning)
    # Get the filename: the lowest directory (where the root files are stored).
    file_name = file.split("/")[-1]
    file_name = file_name.split(".")[:5]
    file_name = "_".join(file_name)
    print("Processing : ", file_name)

    # List of variables
    constituent_vars = Specific_Set4_Parameters.qg_constituent_vars
    jet_vars = Specific_Set4_Parameters.qg_jet_vars
    
    # Get the dataframes from root file
    constituent_pdf = make_df(file, "analysis", constituent_vars)
    jet_pdf = make_df(file, "analysis", jet_vars)
    
    constituent_pdf.reset_index(inplace=True)
    jet_pdf.reset_index(inplace=True)
    
    # Translate MeV distributions into GeV ones
    for var in Specific_Set4_Parameters.vars_convert_MeV_to_GeV_constituent:
        constituent_pdf[var] =  constituent_pdf[var].div(1000)
    for var in Specific_Set4_Parameters.vars_convert_MeV_to_GeV_jet:
        jet_pdf[var] =  jet_pdf[var].div(1000)
    
    constituent_pdf, jet_pdf = event_cleaning(constituent_pdf, jet_pdf)

    dictionnary_result, dictionnary_exception  = perform_clustering("cambridge", float(0.5), jet_pdf, constituent_pdf)
    list_of_jets = dictionnary_result["JuniprJets"]
    test_size = 0.2
    random.shuffle(list_of_jets)
    size_of_data_test = int(len(list_of_jets) * test_size)
    list_train = list_of_jets[size_of_data_test:]
    list_test  = list_of_jets[:size_of_data_test]
    dictionnary_result_train = {"JuniprJets": list_train}
    dictionnary_result_test  = {"JuniprJets": list_test}
    
    os.makedirs(save_path, exist_ok=True)
    
    save_name = os.path.join(save_path, file_name)
    
    save_junipr_data_to_json(dictionnary_exception, save_name+ "_EXCEPTIONS")
    save_junipr_data_to_json(dictionnary_result_train, save_name+"_train")
    save_junipr_data_to_json(dictionnary_result_test, save_name+"_test")
    warnings.filters = original_warnings


def save_junipr_data_to_json(dictionnary, save_name):
    with open(save_name + '.json', "w") as write_file:
            json.dump(dictionnary, write_file)



