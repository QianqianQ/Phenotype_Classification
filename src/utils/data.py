import os
import sys
import fnmatch
import pickle
from argparse import ArgumentParser

import pandas as pd
import numpy as np

project_path = "/scratch/cs/csb/projects/phenotype_classification"
CMV_data_path = os.path.join(project_path, "CMV/data/processed")
RA_data_path = os.path.join(project_path, "RA/data/processed")

def data_load(dataset, file_name, pandas=True):
    """
    Arguments:
        dataset {str} -- the dataset to use, two choices: 'CMV', 'RA'
        file_path {str} -- the data file to load 
    """
    if dataset == 'CMV':
        data_path = CMV_data_path
    elif dataset == 'RA':
        data_path = RA_data_path

    full_file_path = os.path.join(data_path, file_name)
    
    try:
        if fnmatch.fnmatch(full_file_path, '*.pkl') or fnmatch.fnmatch(full_file_path, '*.pickle'):
            if pandas:
                data = pd.read_pickle(full_file_path)
            else:
                with open(full_file_path,'rb') as f:
                    data = pickle.load(f) 
                
        elif  fnmatch.fnmatch(full_file_path, '*.csv'):
            data = pd.read_csv(full_file_path)
        else:
            raise ValueError('Not supported file type')
    except Exception as e:
        return e

    return data


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


def group_data(data_path,files,group,seqStatus,sample_name,output_path):
    
    count_dict = dict()

    for i in range(len(files)):
        try:
            name,counts = get_info(data_path+files[i],group,seqStatus)
        except:
            raise ValueError('Something wrong when reading data file')

        for n, size in zip(name,counts):
            if n not in count_dict: # if the TCR appears the first time on the whole dataset

                # construct a new all-zero array whose length is the number of samples
                count_dict[n] = np.zeros(len(files), dtype=np.uint32) 
                count_dict[n][i] = size # modify the count in the array

            if n in count_dict: # if the TCR has occurred before
                count_dict[n][i] = size

    print('\nThe total number of unique %s on the whole dataset:'%group,len(count_dict))
    if group != 'aminoAcid' and group != 'amino_acid':
        count_df = pd.DataFrame.from_dict(count_dict)
        count_df.insert(0,'sample_name',sample_name)
        count_df.to_pickle(output_path+ group+'.pkl')
    else:
        TCRs = list(count_dict.keys())
        for TCR in TCRs:
            if (count_dict[TCR]==0).sum()>=(len(files)-1): # only one sample has non-zero count
                count_dict.pop(TCR) # pop it from the dict

        print('After dropping the CDR3 occurring in only one sample, '+
            'the total number of unique TCRs on the whole dataset:', len(count_dict))
        print()
        with open(output_path + group + '.pkl', 'wb') as f:
            pickle.dump(count_dict, f, pickle.HIGHEST_PROTOCOL) # save count dict as pkl file