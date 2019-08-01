# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Helpers
import os
import sys
from argparse import ArgumentParser
from scipy.sparse import load_npz

from scipy.stats import fisher_exact
from sklearn.model_selection import LeaveOneOut,KFold,StratifiedKFold

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument('-n','--num',dest='num',type=int,
                        help='the nth sample left out as test sample')

     # Parsing of parameters
    try:
        args = parser.parse_args()
    except:
        # parser.print_help()
        sys.exit(1)

    try:
        n = args.num

    except:
        raise ValueError('Wrong input!')

    data = load_npz('../data/'+'X_sparse/train_count.npz')
    train_y = pd.read_csv('../data/train_Y.csv')
    p_values_origin = pd.read_pickle('../data/inc_p_values.pkl')
    num_TCRs = pd.read_csv('../data/MAP_estimator/train.csv')[['sample_name','unique_TCRs']]
    TCRs = p_values_origin.columns.values

    print('The %dth samples as the test sample'%n)
    flag = 0
    kf = LeaveOneOut()
    for train_index,test_index in kf.split(train_y): # for each cv round
        if flag == n:
            break
        else:
            flag += 1

    test_cv = train_y.iloc[test_index]
    test_sample = test_cv['sample_name'].values[0]
    print('Cross validation test sample:',test_sample)
    p_values_df = pd.read_pickle('../data/LOO_p_values/'+test_sample+'.pkl')
    
    MAP_data = dict()
    MAP_data['sample_name'] = train_y['sample_name']
    MAP_data['phenotype_status'] = train_y['phenotype_status']

    thresholds = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
    for threshold in thresholds:
        print('Threshold:',threshold)
        kept_TCRs = p_values_df.T[p_values_df.T['p_value']<=threshold].T.columns.values
        print('The number of associated TCRs:',len(kept_TCRs))
        indices = [p_values_origin.columns.get_loc(col) for col in kept_TCRs]
        sub_data = data[:,indices]
        counts = np.count_nonzero(sub_data.toarray(),axis=1)
        MAP_data[threshold] = counts

    MAP_df = pd.DataFrame(MAP_data)
    MAP_df = MAP_df.merge(num_TCRs,on='sample_name')

    MAP_df.to_pickle('../data/LOO_MAP_data/'+test_sample+'.pkl')

