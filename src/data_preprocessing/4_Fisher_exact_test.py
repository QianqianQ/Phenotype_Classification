'''
USAGE:

This script generates:
* A dataframe of TCR p-values containing columns: 'phenotype-','phenotype+','p-value'
* Also a pkl file storing the list of indice of TCRs ranking by p-values(smallest to biggest)

Take 2 arguments:
* 2 inputs: 1.TCR incidence dictionary pkl file 2.sample label csv file
* 2 outputs: 1. pkl file storing TCR-p-value df 2. pkl file storing indice of TCRs

Exmple:
python Fisher_exact_test.py -i TCR_inc_dict.pkl sample_label.csv -o TCR_p_value_df.pkl TCRs_ranking_ind.pkl
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


def cal_p_value1(present_c0,present_c1,class_counts,alternative='greater'):
    '''
    present_c0: the number of negative samples where the TCR present
    present_c1: the number of positive samples where the TCR present
    class_counts: the class distribution: [class_negative(0),class_positive(1)]
    alternative: Choose the type of test: greater, less, two_sided

    contingency table: [[present_c1, absent_c1], [present_c0, absent_c0]]
    '''

    absent_c0 = class_counts[0]-present_c0 # the number of negative samples where the TCR absent 
    absent_c1 = class_counts[1]-present_c1 # the number of positive samples where the TCR absent

    # Conducting fisher exact test 
    _, pvalue = fisher_exact([[present_c1, absent_c1], [present_c0, absent_c0]],alternative=alternative)
    return pvalue

def cal_p_value2(row,alternative='greater'):
    '''
    This function calculating Fisher exact test p-value based on pandas dataframe
    '''
    present_c0 = row['incidence in phenotype-']
    absent_c0 = row['absence in phenotype-']
    
    present_c1 = row['incidence in phenotype+']
    absent_c1 = row['absence in phenotype+']

    _, pvalue = fisher_exact([[present_c1, absent_c1], [present_c0, absent_c0]],alternative=alternative)
    return pvalue


if __name__=="__main__":

    start = time.time()

    # Setting argumentParser
    parser = ArgumentParser()
    parser.add_argument('-m','--method',choices=[1,2],dest='method',type=int,default=1,
                        help="Indicate the type of the input: dictionary/dataframe")
    parser.add_argument('-i','--input',nargs=2,dest='input',type=str,
                        help='input files: arg1: TCR incidence dictionary pkl file,\
                                        arg2: sample label csv file')
    parser.add_argument('-t','--type',choices=['pos','neg'],dest='Fisher_type',type=str,default='pos',
                        help="Indicate the type of the Fisher exact test: positive/negative")
    parser.add_argument('-o','--output',nargs=2,dest='output',type=str,
                        default=['inc_and_p_values_df.pkl','TCRs_ranking_ind.pkl'],
                        help='Output files: 1. pkl file storing p-value dataframe, \
                            2. pkl file storing the indices of the TCRs ranked by p-value(lowest to highest)')
    # Parsing of parameters
    try:
        args = parser.parse_args()
    except:
        # parser.print_help()
        sys.exit(1)
    
    try:
        # method used to perform Fisher exact test
        method = args.method
        # TCR incidence dict  
        inc_dict_file = args.input[0]
        if os.path.isfile(inc_dict_file) == False:
            raise FileExistsError('File not found!')
        with open(inc_dict_file,'rb') as f:
            inc_dict = pickle.load(f)
        # sample label file
        sample_label_file = args.input[1]
        if os.path.isfile(sample_label_file) == False:
            raise FileNotFoundError('Label file not found!')
        sample_label = pd.read_csv(sample_label_file)
        # type of the Fisher
        Fisher_type = args.Fisher_type
        # ouput df 
        output_df_file = args.output[0]
        output_ind_file = args.output[1]
    except:
        raise ValueError('Wrong input!')

    # Get class distribution
    class_counts = sample_label['phenotype_status'].value_counts()

    if method == 1:
        # Compute fisher exact test for each TCR
        for tcr in inc_dict:
            if Fisher_type == 'pos':
                p_value = cal_p_value1(inc_dict[tcr][0],inc_dict[tcr][1],class_counts)
            else:
                p_value = cal_p_value1(inc_dict[tcr][0],inc_dict[tcr][1],class_counts,'less')
            inc_dict[tcr].append(p_value)

        inc_p_value_df = pd.DataFrame.from_dict(inc_dict,orient='index') # generate df from dictionary
        inc_p_value_df.columns = ['incidence in phenotype-','incidence in phenotype+','p_value'] # rename df

    else:
        # Convert incidence dict to df: TCR is index
        inc_p_value_df = pd.DataFrame.from_dict(inc_dict,orient='index')
        inc_p_value_df.columns = ['incidence in phenotype-','incidence in phenotype+'] # rename cols
        # Create new cols: absence in phenotype-, absence in phenotype+
        inc_p_value_df['absence in phenotype-'] = class_counts[0]- inc_p_value_df['incidence in phenotype-']
        inc_p_value_df['absence in phenotype+'] = class_counts[1]- inc_p_value_df['incidence in phenotype+']
        # Reorder df
        # cols = ['incidence in phenotype-','absence in phenotype-',
        #         'incidence in phenotype+','absence in phenotype+']
        # inc_p_value_df = inc_p_value_df[cols]
        
        # Compute Fiser exact test 
        if Fisher_type == 'pos':
            inc_p_value_df['p_value'] = inc_p_value_df.apply(cal_p_value2,axis=1)
        else:
            inc_p_value_df['p_value'] = inc_p_value_df.apply(cal_p_value2,alternative='less',axis=1)
    # the indice of TCRs ranking by p-values(smallest to biggest)
    ind = np.argsort(inc_p_value_df['p_value'].values) 

    # Save df as a pkl file
    with open(output_df_file, 'wb') as f:
        pickle.dump(inc_p_value_df, f, pickle.HIGHEST_PROTOCOL)

    # Save indice as a pkl file
    with open(output_ind_file, 'wb') as f:
        pickle.dump(ind, f, pickle.HIGHEST_PROTOCOL)

    end = time.time()
    print("\nComplete! The TCR_incidences_and_p_values dataframe"+ 
        "and TCRs_sorted_indice have been stored!")
    print('Time elapsed %d seconds.' % (end-start))

    