#############################################################################
#
# UprootNTuples.py
#
# Convert the NTuple root file outputs to Numpy arrays and possibly pandas
#
# Author -- Aaron O'Neill (08/04/2020)
# email  -- aaron.aoneill@physics.ox.ac.uk
#
#############################################################################

import uproot as upr
import sys
import os
from optparse import OptionParser
import pprint
import numpy as numpy
import pandas as pd
import utils

def LoadOptions():
    parser = OptionParser()

    parser.add_option("-i", "--inList", dest="in_list", default='', type=str, help='List of plots, root file inputs, that you wish to be converted into tables.' )
    parser.add_option("-o", "--outDir", dest="out_dir", default='', type=str, help='Output directory to store the output files (absolute path!).' )
        
        (options, args) = parser.parse_args()
        options_dict  = vars(options)
        
            return options_dict

def CheckOptions(opt):
    print('Checking your options now...')

    if opt['in_list']=='':
        print('\033[31m ERROR: Please specify an input list containing your inputs using -i or --inList.\033[0;0m')
        quit()
            
            if not os.path.isfile(opt['in_list']):
                print('\033[31m ERROR: Your input list does not seem to exist please check the path.\033[0;0m')
                quit()
                    
                    if not os.path.isdir(opt['out_dir']):
                        print('\033[31m ERROR: Your output directory does not seem to exist, please create it.\033[0;0m')
                        quit()

#Retrieve the list of input RT files to produce the full tables.
def GetInputList():
    input_list = []
    input_file = open(opt['in_list'],'r')
    print('Finding your inputs...')
    for x in input_file.readlines():
        x = x.rstrip('\n')
        if not x.startswith("#"): input_list += [x]
        print(x)
            
            return input_list

def Uproot(input_list):
    #General comment: could make this more generic and split into several functions.
    upr_files = []
    #Create empty dataframe for combined output.
    combined_total_tree_pdf = pd.DataFrame()
    for File in input_list:
upr_file = upr.open(File)
upr_files += [upr_file]
file_name = File.split("/")[-1]

print('I have found the following trees: ')
pprint.pprint(upr_file.keys())

#Will have to cycle systemaics here using generateSyst.py.
nominal_tree = upr_file["Nominal"]
    common_tree = upr_file["commonValues"]
    
    print('Nominal')
    pprint.pprint(nominal_tree.keys())
    print('Common')
    pprint.pprint(common_tree.keys())
    
    #Retrieve the cleaning variables list.
    clean_vars_list = utils.nominal_cleaning_vars + utils.qg_tagging_vars
    
    #Form the data frames
    nominal_tree_pdf = nominal_tree.pandas.df(clean_vars_list)
    pprint.pprint(nominal_tree_pdf)
    common_tree_pdf = common_tree.pandas.df(utils.common_cleaning_vars)
    pprint.pprint(common_tree_pdf)
    
    total_tree_pdf = common_tree_pdf.join(nominal_tree_pdf)
    pprint.pprint(total_tree_pdf)
    
    #Form a cut dict and import this.
    #Apply basic analysis cuts here.
    #Combine this into one operation? As seen here:
    #https://thispointer.com/python-pandas-how-to-drop-rows-in-dataframe-by-conditions-on-column-values/
    total_tree_pdf.drop(total_tree_pdf[total_tree_pdf['numPrimaryVertices'] == 0].index, inplace=True)
    total_tree_pdf.drop(total_tree_pdf[total_tree_pdf['hasBadJet'] == 1].index, inplace=True)
    total_tree_pdf.drop(total_tree_pdf[total_tree_pdf['hasBadMuon'] == 1].index, inplace=True)
    total_tree_pdf.drop(total_tree_pdf[total_tree_pdf['hasCosmic'] == 1].index, inplace=True)
    total_tree_pdf.drop(total_tree_pdf[total_tree_pdf['PVnumTrk'] < 2].index, inplace=True)
    total_tree_pdf.drop(total_tree_pdf[total_tree_pdf['GenFiltHT'] < 600].index, inplace=True)
    
    #Now drop these variables entirely to save space in vmem and disk.
    total_tree_pdf = total_tree_pdf[utils.qg_tagging_vars]
    pprint.pprint(total_tree_pdf)
    
    #More analysis oreintated cuts, 'baseline jets'.
    total_tree_pdf.drop(total_tree_pdf[total_tree_pdf['jetPt'] < 20].index, inplace=True)
    total_tree_pdf.drop(total_tree_pdf[total_tree_pdf['jetEta'].abs() > 2.5].index, inplace=True)
    
    #Get rid of nasty values in the BDT and truth information
    total_tree_pdf.drop(total_tree_pdf[total_tree_pdf['BDTScore'] == -666.0].index, inplace=True)
    
    #Other data quality cuts.
    #Remove tracks with negative width.
    total_tree_pdf.drop(total_tree_pdf[total_tree_pdf['jetTrackWidthPt500'] < 0.0].index, inplace=True)
    total_tree_pdf.drop(total_tree_pdf[total_tree_pdf['jetTrackWidthPt1000'] < 0.0].index, inplace=True)
    
    #Remove the truth jets with -1 tag to keep the classification binary.
    total_tree_pdf.drop(total_tree_pdf[total_tree_pdf['isTruthQuark'] < 0.0].index, inplace=True)
    
    #Have a quick look at the output.
    pprint.pprint(total_tree_pdf)
    
    #Save the dataframe to csv format.
    total_tree_pdf.to_csv(os.path.join(opt['out_dir'], file_name + '.csv'))
    
    #Running total of the files (this may not be a good idea).
    combined_total_tree_pdf = combined_total_tree_pdf.append(total_tree_pdf)
        
        #Save the combined pandas df to csv
        combined_total_tree_pdf.to_csv(os.path.join(opt['out_dir'], file_name + '_all.csv'))


def main():
    global opt
        opt = LoadOptions()
        CheckOptions(opt)
        input_list = GetInputList()
        Uproot(input_list)
        
            print('Completed successfully.')

if __name__ == "__main__":
    main()
