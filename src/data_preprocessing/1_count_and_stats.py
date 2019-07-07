'''
USAGE:

This script generates:
* A dictionary of TCR counts: key:TCR, values: an array(len(samples)) storing the count per sample
* A statistics csv file: number of total TCRs, number of unique TCRs per sample

Take 3 arguments:
* data file path
* a list of column names extracted from file (it could be modified by users based on their own needs, and a default value is provided)
  Note: The last parameter must be the frameType(sequenceStatus)
  e.g. amino_acid v_family v_gene v_allele j_family j_gene j_allele frame_type
* 2 output file names: 1.a pkl file storing count dict 2. a pkl file storing count dataframe  3.a csv file storing stat

Exmple:
python count_and_stats.py -p train_data/ -l amino_acid v_family v_gene v_allele j_family j_gene j_allele frame_type -o count_dict.pkl count_df.pkl samples_stats.csv
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

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


def get_info(file,cols):
    ''' 
    Get number of total TCRs,number of unique TCRs, list of TCRs and their counts from a sample file
    '''
    try:
        # Read file only with needed cols
        df_temp = pd.read_csv(file, delimiter='\t',usecols=cols,dtype='str')  

        # Get sample name (depends on the data format)
        # sample_name = df_temp['sample_name'][0] 

        df_temp = df_temp[df_temp[cols[-1]] == 'In']  # only the sequences with sequence_status==In are kept

        # Grouping CDR3,V,J to form TCRs
        grouped = df_temp.fillna('null').groupby(cols[:-1]) 
        
        # Total number of unique TCRs
        num_uniq_TCRs = len(grouped.groups.keys())

        # Count of each TCR 
        grouped_df = grouped.size().reset_index(name='size')

        # Total number of TCRs
        num_total_TCRs = sum(grouped_df['size']) 

        # List of TCRs (combine CDR3,V,J as tuple)
        TCRs_temp = [tuple(row[col] for col in cols[:-1]) for _, row in grouped_df.iterrows()]

        counts = grouped_df['size'].tolist()
    except:    
        raise ValueError

    return num_total_TCRs,num_uniq_TCRs,TCRs_temp,counts


if __name__=="__main__":

    start = time.time()  

    # Setting argumentParser
    parser = ArgumentParser()
    parser.add_argument('-p','--path',dest='path',type=str,help='data path reading the data files')
    parser.add_argument('-l','--list',dest='cols',type=str,nargs='*',
                        # default=['amino_acid', 'v_family', 'v_gene', 
                        #         'v_allele', 'j_family','j_gene', 'j_allele','frame_type'],
                        default=['aminoAcid', 'vFamilyName', 'vGeneName', 'vGeneAllele', 'jFamilyName',
                                'jGeneName', 'jGeneAllele', 'sequenceStatus'],
                        help='Names of needed columns')
    parser.add_argument('-o','--output',nargs=3,dest='output',type=str,
                        default=['count_dict.pkl','count_df.pkl','samples_stats.csv'],
                        help='Output files: arg1: pkl file storing frequency dictionary,\
                            arg2:pkl file storing frequency dataframe,\
                            arg3:csv file storing number of total TCRs and number of unique TCRs per sample')

    # Parsing of parameters
    try:
        args = parser.parse_args()
    except:
        # parser.print_help()
        sys.exit(1)
    
    try:
        # dataset path  
        path = args.path
        if os.path.isdir(path) == False:
            raise FileExistsError('No such directory!')
        # needed columns
        needed_cols = args.cols
        if len(needed_cols)<=1:
            raise ValueError('Too few columns extracted')
        # ouput files
        output_dict = args.output[0] 
        output_df = args.output[1]
        output_stat = args.output[2]
    except:
        raise ValueError('Wrong input!')

    # List of file names
    files = []
    for f in os.listdir(path):
        if f[-4:]== '.tsv': # Check whether it is tsv file
            files.append(f)
    if len(files)==0:
        raise FileExistsError('No data files found!')
    print('The number of samples:',len(files))
    print()
    # with open('file_names.pkl', 'wb') as f:
    #         pickle.dump(files, f, pickle.HIGHEST_PROTOCOL) # save files names as pkl file

    # init count dictionary: key is TCR, value is an array(len(files)) storing count of the TCR in per sample  
    count_dict = dict()
    # init dict of total number of TCRs: key is sample name, value is the total number of TCRs
    num_total_TCRs_dict = dict()
    # init dict of total number unique TCRs: key is sample name, value is the total number of unique TCRs
    num_uniq_TCRs_dict = dict()
    # a list of sample names
    sample_name = []

    # Get total number of TCRs, total number of unique TCRs, TCRs and their counts in per sample,
    # and then add them to the dicts
    for i in tqdm(range(len(files))):
        try:
            num_total_TCRs,num_uniq_TCRs,TCRs_temp,counts = get_info(path+files[i],needed_cols)
        except:
            raise ValueError('Something wrong when reading data file')
        i_sample_name = files[i].replace(files[i][-4:],'') # remove suffix from the file name as sample name
        num_total_TCRs_dict[i_sample_name] = num_total_TCRs
        num_uniq_TCRs_dict[i_sample_name] =num_uniq_TCRs
        sample_name.append(i_sample_name)

        # Add TCRs and TCRs' counts to count dict
        for TCR, size in zip(TCRs_temp,counts):
            if TCR not in count_dict: # if the TCR appears the first time on the whole dataset

                # construct a new all-zero array whose length is the number of samples
                count_dict[TCR] = np.zeros(len(files), dtype=np.uint16) 
                count_dict[TCR][i] = size # modify the count in the array

            if TCR in count_dict: # if the TCR has occurred before
                count_dict[TCR][i] = size

    print('\nThe total number of unique TCRs on the whole dataset:',len(count_dict))

    # Dropping the TCRs appearing in only one sample
    TCRs = list(count_dict.keys())
    for TCR in TCRs:
        if (count_dict[TCR]==0).sum()>=(len(files)-1): # only one sample has non-zero count
            count_dict.pop(TCR) # pop it from the dict

    print('After dropping the TCRs occurring in only one sample, '+
        'the total number of unique TCRs on the whole dataset:', len(count_dict))
    print()

    # Generate statistics files
    try:
    # convert num_total_TCRs, num_uniq_TCRs dicts to dataframes
        total_TCRs_df = pd.DataFrame.from_dict(num_total_TCRs_dict, orient='index')
        uniq_TCRs_df = pd.DataFrame.from_dict(num_uniq_TCRs_dict, orient='index')
        # reset sample_name as a column
        total_TCRs_df.reset_index(inplace=True)
        uniq_TCRs_df.reset_index(inplace=True)
        
        # rename columns
        total_TCRs_df.columns = ['sample_name','total_TCRs']
        uniq_TCRs_df.columns = ['sample_name','unique_TCRs']
        # merge two df
        stat_df = total_TCRs_df.merge(uniq_TCRs_df,on='sample_name')
        # save df as csv
        stat_df.to_csv(output_stat,index=False)
        print('Save statistics file successfully!')

        # save count dict as pkl file
        with open(output_dict, 'wb') as f:
            pickle.dump(count_dict, f, pickle.HIGHEST_PROTOCOL) # save count dict as pkl file
        print('Save count dict as pkl file successfully!')

        # save count dict as count df
        try:
            count_df = pd.DataFrame.from_dict(count_dict)
            count_df.insert(0,'sample_name',sample_name)
        except:
            raise Exception('Error occurs when converting count dict to df')
        # save count df as pkl file
        try:
            with open(output_df, 'wb') as f:
                pickle.dump(count_df, f, pickle.HIGHEST_PROTOCOL) # save count df as pkl file
            print('Save count daraframe as pkl file successfully!')
        except:
            print('Error occurs when save count df as pkl file, try to save as hdf5 format')
            try:
                store = pd.HDFStore('storage.h5')
                count_df.to_hdf(store,'count_df')
                store.close()
            except:
                raise Exception('Failed to save count dataframe!')
    except:
        raise ValueError('Problem occurs when outputting files')

    end = time.time()
    print("\nComplete! Time elapsed %d seconds." % (end-start))

    
