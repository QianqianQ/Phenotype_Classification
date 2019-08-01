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

def cal_p(col,class_counts):
    present_c0 = col.loc['CMV-']
    present_c1 = col.loc['CMV+']
    
    absent_c0 = class_counts[0]-present_c0 # the number of negative samples where the TCR absent 
    absent_c1 = class_counts[1]-present_c1 # the number of positive samples where the TCR absent

    _, pvalue = fisher_exact([[present_c1, absent_c1], [present_c0, absent_c0]],alternative='greater')
    return pvalue

def TCRs_selection_CV(train_data, test_index, train_y,p_values_origin, class_counts, threshold):
    TCRs = p_values_origin.columns.values
    p_values_df = p_values_origin.copy()
    for test_ind in test_index:
        test_cv = train_y.iloc[test_ind]
        non_zero_ind = train_data[test_ind,:].nonzero()[1]
        if test_cv['phenotype_status'] == 0:
            p_values_df.loc['CMV-',TCRs[non_zero_ind]] -= 1
        elif test_cv['phenotype_status'] == 1:
            p_values_df.loc['CMV+',TCRs[non_zero_ind]] -= 1
        else:
            raise ValueError
        p_values_df.loc['p_value'][TCRs[non_zero_ind]] = p_values_df[TCRs[non_zero_ind]].apply(cal_p,class_counts=class_counts,axis=0)
        
    sub_p_values = p_values_df[TCRs[p_values_df.loc['p_value']<=threshold]]
    sub_p_values.drop(['CMV+','CMV-'],axis=0,inplace=True)
    return sub_p_values

if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument('-n','--num',dest='num',type=int,
                        help='the nth round')

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
    p_values_df = pd.read_pickle('../data/inc_p_values.pkl')

    print('The %dth samples as the test sample'%n)
    flag = 0
    # kf = LeaveOneOut()
    kf = StratifiedKFold(10,random_state=0)
    for train_index,test_index in kf.split(train_y): # for each cv round
        if flag == n:
            break
        else:
            flag += 1

    train_cv = train_y.iloc[train_index]
    class_counts = train_cv['phenotype_status'].value_counts()
    
    test_cv = train_y.iloc[test_index]

    sub_p_values = TCRs_selection_CV(data,test_index,train_y,p_values_df,class_counts,0.1)

    test_sample = test_cv['sample_name'].values[0]
    print('Cross validation test sample:',test_sample)
    sub_p_values.to_pickle('../data/LOO_p_values/'+test_sample+'.pkl')

