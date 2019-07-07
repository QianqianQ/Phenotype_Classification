'''
USAGE:

To get TCR incidences in phenotype-negative and phenotype-positive subjects, preparing for calculating p-values

This script generates:
* A dictionary of TCR incidences: key:TCR, values: incidences within two classes: [incidence in c_0, incidence in c_1]

Take 2 arguments:
* count df file (contaning 'phenotype_status' column)
* 1 output file: dictionary storing TCR incidences
Exmple:
python 3_TCRs_inc_dict.py -t dict -i count_dict.pkl sample_label.csv -o TCR_inc_dict.pkl
'''
#==============================================================================================================================================================
__author__ = 'Qin Qianqian'
__email__ = 'qianqian.qin@outlook.com'
__status__ = 'Development'
#==============================================================================================================================================================

# Libs
import numpy as np
import pandas as pd
from sklearn.externals import joblib

import os
import sys
from argparse import ArgumentParser

import time
import pickle

import warnings
warnings.filterwarnings('ignore')


if __name__=="__main__":

    start = time.time()

    # Setting argumentParser
    parser = ArgumentParser()
    parser.add_argument('-t','--type',choices=['dict','df'],dest='input_type',type=str,default='dict',
                        help="Indicate the type of the input: dictionary/dataframe")
    parser.add_argument('-i','--input',nargs=2,dest='input',type=str,
                        help='input files: arg1.TCR count dict/ TCR count dataframe/ TCR bin dataframe, \
                                        arg2: sample label csv file')
    parser.add_argument('-o','--output',dest='output',type=str,default='TCR_incidence_dict.pkl',
                        help='Output file: pkl file storing incidence dictionary')

    # Parsing of parameters
    try:
        args = parser.parse_args()
    except:
        # parser.print_help()
        sys.exit(1)
    
    try:
        # input type
        input_type = args.input_type
        # input file        
        if input_type == 'dict':
            count_dict_file = args.input[0]
            if os.path.isfile(count_dict_file) == False:
                raise FileNotFoundError('Count dictionary not found!')
            with open(count_dict_file,'rb') as f:
                count_dict = pickle.load(f)
        else:
            df_file = args.input[0]
            if os.path.isfile(df_file) == False:
                raise FileNotFoundError('Count dataframe not found!')
            data_df = pd.read_pickle(df_file)
        # input: sample labels
        sample_label_file = args.input[1]
        if os.path.isfile(sample_label_file) == False:
            raise FileNotFoundError('Label file not found!')
        sample_label = pd.read_csv(sample_label_file)
        # ouput: TCR incidence dict
        output_dict = args.output
    except:
        raise ValueError('Wrong input!')

    # init TCR incidence dict
    TCRs_inc = dict()

    if input_type == 'df':
        try:
        # the subdf of different phenotype_status(negative/positive)
            neg_df = data_df[data_df['phenotype_status']==0]
            pos_df = data_df[data_df['phenotype_status']==1]

        except KeyError:
            print('The dataframe does not contain label column!')
            # the indice of different phenotype_status(negative/positive) in the sample_label df (same as count array)
            # neg_ind = sample_label[sample_label['phenotype_status']==0].index.tolist()
            # pos_ind = sample_label[sample_label['phenotype_status']==1].index.tolist()
            # neg_df = data_df.iloc[neg_ind]
            # pos_df = data_df.iloc[pos_ind]
            neg_samples = sample_label[sample_label['phenotype_status']==0]['sample_name']
            pos_samples = sample_label[sample_label['phenotype_status']==1]['sample_name']
            neg_df = data_df[data_df['sample_name'].isin(neg_samples)]
            pos_df = data_df[data_df['sample_name'].isin(pos_samples)]

        print('The number of phenotype-negative samples:',neg_df.shape[0])
        print('The number of phenotype-positive samples:',pos_df.shape[0])

        # the list of TCRs
        try:
            TCRs = data_df.drop(['sample_name','phenotype_status'],axis=1).columns.values
        except KeyError:
            TCRs = data_df.drop(['sample_name'],axis=1).columns.values

        for tcr in TCRs:
            # the number of samples where the TCR occurs in the negative subdf
            # neg_num = len(np.nonzero(neg_df[tcr].values)[0]) 
            neg_num = np.count_nonzero(neg_df[tcr].values)
            
            # the number of samples where the TCR occurs in the positive subdf
            # pos_num = len(np.nonzero(pos_df[tcr].values)[0]) 
            pos_num = np.count_nonzero(pos_df[tcr].values)
            
            TCRs_inc[tcr] = [neg_num,pos_num] # add the TCR incidence to dict
    else:
        # the indice of different phenotype_status(negative/positive) in the sample_label df (same as count array)
        neg_ind = sample_label[sample_label['phenotype_status']==0].index.tolist()
        pos_ind = sample_label[sample_label['phenotype_status']==1].index.tolist()
        
        print('The number of phenotype-negative samples:',len(neg_ind))
        print('The number of phenotype-positive samples:',len(pos_ind))

        # Generate TCR incidence dict
        for TCR in count_dict: # for each TCR
            # get the array of count: length is the size of samples, each element is the TCR count of each sample 
            count = count_dict[TCR] 
            neg = np.take(count,neg_ind) # subarray: TCR count of negative samples
            # neg_num = len(np.nonzero(neg)[0]) # the number of negative samples where the TCR occurs
            neg_num = np.count_nonzero(neg)
            pos = np.take(count,pos_ind) # subarray: TCR count of positive samples
            pos_num = np.count_nonzero(pos) # the number of positive samples where the TCR occurs
            TCRs_inc[TCR] = [neg_num,pos_num] # add the TCR incidence to dict

    # Save TCR incidence dict as pkl file
    with open(output_dict, 'wb') as f:
        pickle.dump(TCRs_inc, f, pickle.HIGHEST_PROTOCOL)

    end = time.time()
    print('\nComplete! The TCR incidence dictionary has been stored!')
    print('Time elapsed %d seconds.' % (end-start))