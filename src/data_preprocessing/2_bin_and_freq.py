'''
USAGE:

This script generates:
* Binary version dataframe from df of count version
* Frequency version dataframe from df of count version. 
Note: If could not be saved as df, corresponding sparse matrice will be generated

Take 2 arguments:
* 2 input files: 1. count df/dict 2. TCRs_stas csv file(contain number of total TCRs per sample)
* 2 output files: 1.binary version df pkl file 2. frequency version df pkl file
Exmple:
python bin_and_freq.py -t df -i count_df.pkl TCRs_stats.csv -o bin_df.pkl freq_df.pkl
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
from tqdm import tqdm
import pickle
from scipy import sparse

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


if __name__=="__main__":

    start = time.time()    

    # Setting argumentParser
    parser = ArgumentParser()
    parser.add_argument('-t','--type',choices=['df','dict'],dest='input_type',type=str,default='df',
                        help="Indicate the type of the input: dataframe/dictionary")
    parser.add_argument('-i','--input',nargs=2,dest='input',type=str,
                        help='input files: arg1: count df file or count dict file (pkl format),\
                                        arg2: TCRs stats csv file')
    parser.add_argument('-o','--output',nargs=2,dest='output',type=str,
                        default=['bin_df.pkl','freq_df.pkl'],
                        help='Output pkl files: 1. Binary version df 2. Frequency version df')
    # Parsing of parameters
    try:
        args = parser.parse_args()
    except:
        # parser.print_help()
        sys.exit(1)
    
    try:
        # input type
        input_type = args.input_type
        # count file
        count_file = args.input[0]
        if os.path.isfile(count_file) == False:
            raise FileExistsError('Count file not found!')
        if input_type == 'df':
            count_df = pd.read_pickle(count_file)
        else:
            with open(count_file,'rb') as f:
                count_dict = pickle.load(f) 
        # TCRs stats file
        TCRs_stats_file = args.input[1]
        if os.path.isfile(TCRs_stats_file) == False:
            raise FileNotFoundError('TCR stats file not found!')
        TCRs_stats = pd.read_csv(TCRs_stats_file)
        # ouput df 
        bin_df_file = args.output[0]
        freq_df_file = args.output[1]
    except:
        raise ValueError('Wrong input!')

    if input_type == 'df':
        # The list of TCRs
        try: # handling the count df containing the 'phenotype_status' column
            TCRs = count_df.drop(['sample_name','phenotype_status'],axis=1).columns.values
        except KeyError: # handling the count df without the 'phenotype_status' column
            print('The dataframes does not contain the phenotype_status column')
            TCRs = count_df.drop(['sample_name'],axis=1).columns.values
        else:
            raise Exception('An unexpected error occurred')

        # Binary version
        sample_name = count_df['sample_name'] # extract sample_name column
        # # extract phenotype_status column
        # try:
        #     phenotype_status = count_df['phenotype_status'] 
        # except:
        #     print()
        TCRs_matrix = count_df[TCRs].copy().values # matrix of TCR counts
        bin_df = pd.DataFrame(TCRs_matrix,columns=TCRs) # generate binary df
        bin_df[bin_df!=0]=1 # replace non-zero values as 1
        bin_df.insert(0,'sample_name',count_df['sample_name']) # insert sample_name
        # # insert phenotype_status
        # try:
        #     bin_df.insert(1,'phenotype_status',count_df['phenotype_status']) 
        # except:
        #     print()

        # Frequency version, from count df directly
        try:
            TCRs_stats.drop(['unique_TCRs','phenotype_status'],axis=1,inplace=True)
        except KeyError:
            TCRs_stats.drop(['unique_TCRs'],axis=1,inplace=True)

        freq_df = TCRs_stats.merge(count_df,on='sample_name')
        for tcr in TCRs:
            freq_df[tcr] = freq_df[tcr].div(freq_df['total_TCRs'],axis=0)

        freq_df.drop(['total_TCRs'],axis=1,inplace=True) 

    else:
        # Binary version
        bin_df = pd.DataFrame.from_dict(count_dict) # convert count dict to df
        bin_df[bin_df!=0]=1 # replace non-zero values as 1
        bin_df.insert(0,'sample_name',TCRs_stats['sample_name']) # insert sample_name

        # Frequency version
        freq_dict = dict() # init frequency dict
        total_TCRs = TCRs_stats['total_TCRs'].values # series of total_TCRs
        for tcr in count_dict:
            count = count_dict[tcr] # counts of tcr 
            freq = count / total_TCRs # count is divded by the total number of TCRs
            freq_dict[tcr] = freq # add to freq_dict
        
        freq_df = pd.DataFrame.from_dict(freq_dict) # convert freq dict to df
        freq_df.insert(0,'sample_name',TCRs_stats['sample_name']) # insert sample_name
        # freq_df.insert(1,'total_TCRs',TCRs_stats['total_TCRs']) # insert total_TCRs

    # Save binary version df as pkl file, if error occur, try to save as hdf5 format or sparse matrix
    try:
        with open(bin_df_file, 'wb') as f:
            pickle.dump(bin_df, f, pickle.HIGHEST_PROTOCOL) 
        print('Save binary daraframe as pkl file successfully!')
    except:
        print('Error occurs when save bin df as pkl file, try to save as hdf5 format')
        try: 
            store = pd.HDFStore('storage.h5')
            bin_df.to_hdf(store,'bin_df')
            store.close()
            print('Save binary daraframe as hdf5 format successfully!')
        except:
            print('Error occurs when save binary df as hdf5 format, try to save as sparse matrix')
            bin_df.drop(['sample_name'],axis=1,inplace=True)
            X_sparse = sparse.csr_matrix(bin_df.as_matrix())
            sparse.save_npz('X_bin.npz', X_sparse)
            print('Save frequency version data as sparse matrix successfully!')

    # Save frequency version df as pkl file, if error occur, try to save as hdf5 format or sparse matrix
    try:
        with open(freq_df_file, 'wb') as f:
            pickle.dump(freq_df, f, pickle.HIGHEST_PROTOCOL) 
        print('Save frequency daraframe as pkl file successfully!')
    except:
        print('Error occurs when save bin df as pkl file, try to save as hdf5 format')
        try: 
            store = pd.HDFStore('storage.h5')
            freq_df.to_hdf(store,'freq_df')
            store.close()
            print('Save frequency daraframe as hdf5 format successfully!')
        except:
            print('Error occurs when save frequency df as hdf5 format, try to save as sparse matrix')
            freq_df.drop(['sample_name'],axis=1,inplace=True)
            X_sparse = sparse.csr_matrix(freq_df.as_matrix())
            sparse.save_npz('X_freq.npz', X_sparse)
            print('Save frequency version data as sparse matrix successfully!')

    end = time.time()
    print("\nComplete! Time elapsed %d seconds." % (end-start))

    