#############################################################################
#
# UpRootLoader.py
#
# A data loader for a given directory of Root files. Loads everything, based on a selection of variables.
#
# Code heavily inspired by: Aaron O'Neill, https://github.com/aponeill89/MLForQgTaggingAndRPVSUSY/.
#
# Author -- Maxence Draguet (21/05/2020)
#
# This loader takes directly the root files and convert them into the expected
# format using UpRoot.
#
# A list of variables of interest should be specified in its constructor.
#
#############################################################################

import os
import sys
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
import uproot as upr
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import fnmatch

from Utils import Specific_Set2_Parameters

class UpRootLoader(ABC):
    def __init__(self, config: Dict):
        self.extract_parameters()
        self.get_inputs_list()
    
    def extract_parameters(self, config: Dict):
        self.data_path = config.get(["relative_data_path"]) #in this case it should be a directory
        self.seed = config.get(["seed"])
        self.test_size = config.get(["BDT_model", "test_size"])
    
    def get_inputs_list(self, list_files)->None:
        """
        Given a list of files extract a list of root files, except if they are commented in the list
        This does not seem appropriate for a directory-based loading (take all root files from data_path).
        """
        self.inputs_list = []
        input_file = open(list_files,'r')
        print('Finding your inputs...')
        for x in input_file.readlines():
            x = x.rstrip('\n')
            if not x.startswith("#"): self.inputs_list += [x]
            print(x)

    def make_df(self, input_file, tree_name, branches):
        """
        Turn the inputs into a panda dataframe (list?)
        """
        print('Getting Tree:', tree_name)
        print('With branches:', branches)
        dataframes = pd.DataFrame()
        for array in tqdm(upr.iterate(input_file + "*.root", tree_name, branches, outputtype=pd.DataFrame, flatten=True),total=len(fnmatch.filter(os.listdir(input_file), '*.root'))):
            dataframes = dataframes.append(array)
        return dataframes

    def event_cleaning(self, pdf):
        """
        Form a cut dict and import this.
        Apply basic analysis cuts here.
        #Combine this into one operation? As seen here:
        #https://thispointer.com/python-pandas-how-to-drop-rows-in-dataframe-by-conditions-on-column-values/
        """
        pdf.drop(pdf[pdf['numPrimaryVertices'] == 0].index, inplace=True)
        pdf.drop(pdf[pdf['hasBadJet'] == 1].index, inplace=True)
        pdf.drop(pdf[pdf['hasBadMuon'] == 1].index, inplace=True)
        pdf.drop(pdf[pdf['hasCosmic'] == 1].index, inplace=True)
        pdf.drop(pdf[pdf['PVnumTrk'] < 2].index, inplace=True)
        pdf.drop(pdf[pdf['GenFiltHT'] < 600].index, inplace=True)

        #Now drop these variables entirely to save space in vmem and disk.
        pdf = pdf[utils.qg_tagging_vars]
        pprint.pprint(pdf)

        #More analysis oreintated cuts, 'baseline jets'.
        pdf.drop(pdf[pdf['jetPt'] < 20].index, inplace=True)
        pdf.drop(pdf[pdf['jetEta'].abs() > 2.5].index, inplace=True)

        #Get rid of nasty values in the BDT and truth information
        pdf.drop(pdf[pdf['BDTScore'] == -666.0].index, inplace=True)

        #Other data quality cuts.
        #Remove tracks with negative width.
        pdf.drop(pdf[pdf['jetTrackWidthPt500'] < 0.0].index, inplace=True)
        pdf.drop(pdf[pdf['jetTrackWidthPt1000'] < 0.0].index, inplace=True)

        #Remove the truth jets with -1 tag to keep the classification binary.
        pdf.drop(pdf[pdf['isTruthQuark'] < 0.0].index, inplace=True)
        
        #Fix the variables given in MeV
        pdf['jetSumTrkPt500'] = pdf['jetSumTrkPt500'].div(1000)
        pdf['jetSumTrkPt1000'] = pdf['jetSumTrkPt1000'].div(1000)
        pdf['jetEnergy'] = pdf['jetEnergy'].div(1000)

        return pdf
    
    def uproot(input_list):
        #General comment: could make this more generic and split into several functions.
        upr_files = []
        
        for File in input_list:
            #Create empty dataframe for combined output.
            #combined_pdf = pd.DataFrame()
            file_name = File.split("/")[-2]
            print(file_name)

            #List of variables for cleaning
            clean_vars_list = utils.nominal_cleaning_vars + utils.qg_tagging_vars

            #Get the dataframes from root file
            common_tree_pdf = MakeDF(File, "commonValues", utils.common_cleaning_vars)
            #Need a more general way of doing this so it agrees with multiIndex dfs.
            common_tree_pdf.index.names = ['entry']
            nominal_tree_pdf = MakeDF(File, "Nominal", clean_vars_list)

            pprint.pprint(common_tree_pdf)
            pprint.pprint(nominal_tree_pdf)
            combined_pdf = nominal_tree_pdf.join(common_tree_pdf)
            pprint.pprint(combined_pdf)

            combined_pdf = EventCleaning(combined_pdf)

            #Save the combined pandas df to csv
            print('Writing output file...')
            combined_pdf.to_csv(os.path.join(opt['out_dir'], file_name + '_all.csv'))
            
        if opt['diagnositc_plots']:
            DiagnosticPlots(combined_pdf)
                
    def save(self):
        pass

