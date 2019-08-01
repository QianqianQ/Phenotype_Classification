"""
This script helps to load needed data
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

CMV_data_path = '../data/processed/'

def load_sparse(version):
    """Load CMV dataset
    
    Arguments:
        version {str} -- The version of data to load: 'binary', 'count' or 'frequency'
    
    Returns:
        train_X, test_X -- sparse matrice of features
        train_y, test_y -- labels of the data
    """
    if version == 'binary':
        train_X = load_npz(CMV_data_path + 'train_bin.npz')
        test_X = load_npz(CMV_data_path + 'test_bin.npz')
    elif version == 'count':
        train_X = load_npz(data_path + 'train_count.npz')
        test_X = load_npz(data_path + 'test_count.npz')
    elif version == 'frequency':
        train_X = load_npz(data_path + 'train_freq.npz')
        test_X = load_npz(data_path + 'test_freq.npz')
    
    else:
        raise ValueError('Wrong input!')
         
    train_y = pd.read_csv(data_path + 'train_Y.csv')['CMV_status']
    test_y = pd.read_csv(data_path + 'test_Y.csv')['CMV_status']

    return train_X,train_y,test_X,test_y


