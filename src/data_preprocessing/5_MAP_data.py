'''
USAGE:

This script generates:
* A dataframe of data needed by MAP classifier: contaning data related to 'total_unique_TCRs' and 'phenotype-associated TCRs' 

Take 2 arguments:
* 2 inputs: 1.TCR p-value df file 2.count df / binary df / count dict 3.TCRs_stats csv file
* 1 parameter: p-value threshold(s)
* output: 1. pkl file storing TCR-p-value df 2. pkl file storing indice of TCRs

Exmple:
python MAP_data_from_df.py -t df -i inc_and_p_values_df.pkl count_df.pkl TCRs_stats.csv -p 0.1 0.2 0.3 -o MAP_data.csv
'''
#==============================================================================================================================================================
__author__ = 'Qin Qianqian'
__email__ = 'qianqian.qin@outlook.com'
__status__ = 'Development'
#==============================================================================================================================================================

# Libs
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

import os
import sys
import time
import pickle
from argparse import ArgumentParser

import warnings
warnings.filterwarnings('ignore')


if __name__=="__main__":

    start = time.time()  

    # Setting argumentParser
    parser = ArgumentParser()
    parser.add_argument('-t','--type',choices=['df','dict'],dest='input_type',type=str,default='df',
                        help="Indicate the type of the second input file: dataframe/dictionary")
    parser.add_argument('-i','--input',nargs=3,dest='input',type=str,
                        help='input files: arg1: TCR p-value df,\
                                        arg2: count df/ binary df/ count dict,\
                                        arg3: TCRs stats csv file')
    parser.add_argument('-p','--parameter',nargs='*',dest='parameter',type=float,
                        help='p-value threshold(s)')
    parser.add_argument('-o','--output',dest='output',type=str,default='MAP_data.csv',
                        help='Output file: MAP data csv file')
    # Parsing of parameters
    try:
        args = parser.parse_args()
    except:
        # parser.print_help()
        sys.exit(1)
    
    try:
        # input type
        input_type = args.input_type
        # TCR p-value df  
        p_value_df_file = args.input[0]
        if os.path.isfile(p_value_df_file) == False:
            raise FileExistsError('p_value df file not found!')
        p_value_df = pd.read_pickle(p_value_df_file)
        # count df / binary df / count dict
        if input_type == 'df':
            df_file = args.input[1]
            if os.path.isfile(df_file) == False:
                raise FileNotFoundError('Count/Binary df file not found!')
            data_df = pd.read_pickle(df_file)
        else:
            count_dict_file = args.input[1]
            if os.path.isfile(count_dict_file) == False:
                raise FileNotFoundError('count dict file not found!')
            with open(count_dict_file,'rb') as f:
                count_dict = pickle.load(f)
        # TCR stats csv file
        TCR_stats_file = args.input[2]
        if os.path.isfile(TCR_stats_file) == False:
            raise FileNotFoundError('TCR stats file not found!')
        TCRs_stats = pd.read_csv(TCR_stats_file)
        # p-value threshold(s)
        thresholds = args.parameter
        # ouput df 
        output_df_file = args.output
    except:
        raise ValueError('Wrong input!')

    
    # Get the number of phenotype-associated TCRs in each sample under different p-value threshold

    # Init phenotype-associated dictionary: key is threshold, 
    # value is a list of numbers of phenotype-associated TCRs in samples
    phenotype_asso_dict = dict() 
    sample_name = TCRs_stats['sample_name'] # list of sample names

    for threshold in thresholds: # for each threshold
        # Get list of TCRs whose p-values smaller than threshold (phenotype-associated)
        TCRs = p_value_df[p_value_df.p_value<=threshold].T.columns.values
        print('threshold:',threshold,', the number of phenotype-asso TCRs:',len(TCRs))

        count_pheno_asso = [] # init a list to store number of phenotype-associated TCRs per sample

        if input_type == 'dict':
            temp_dict = { tcr: count_dict[tcr] for tcr in TCRs } # get count subdict of phenotype-associated TCRs
            data_df = pd.DataFrame(temp_dict) # convert subdict to subdf
            data_df.insert(0,'sample_name',sample_name) # insert 'sample_name'
        
        for i in range(len(sample_name)):
            temp = data_df.loc[data_df.sample_name==sample_name[i]] # row of specific sample in count/bin df
            i_pheno_asso = np.count_nonzero(temp[TCRs].values) # number of phenotype-associated TCRs in the sample
            count_pheno_asso.append(i_pheno_asso)   # add to the list
        phenotype_asso_dict[threshold] = count_pheno_asso # add the list to the dict

    pheno_asso_df = pd.DataFrame(phenotype_asso_dict) # convert phenotype-associated dict to df
    pheno_asso_df.insert(0,'sample_name',sample_name) # insert 'sample_name'

    TCRs_stats = TCRs_stats.merge(pheno_asso_df,on='sample_name') # merge 2 df as a complete MAP data df

    TCRs_stats.to_csv(output_df_file,index=False) # save the data as csv file

    end = time.time()
    print("\nComplete! The MAP data has been stored!")
    print("Time elapsed %d seconds." % (end-start))

    