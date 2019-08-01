# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Helpers
import os
import sys
from argparse import ArgumentParser
from scipy.sparse import load_npz

from scipy.stats import fisher_exact
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import LeaveOneOut,KFold,StratifiedKFold

# Algorithm
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument('-n','--num',dest='num',type=int,
                        help='p-value N Best')

     # Parsing of parameters
    try:
        args = parser.parse_args()
    except:
        # parser.print_help()
        sys.exit(1)

    try:
        NBest = args.num

    except:
        raise ValueError('Wrong input!')

    data_path = '../data/'

    train_X = load_npz(data_path + 'X_sparse/'+ 'train_bin.npz')
    train_y = pd.read_csv(data_path + 'train_Y.csv')

    p_values_origin = pd.read_pickle('../data/inc_p_values.pkl')

    clf = LogisticRegression()

y_true = []
y_proba = []

print('Number of NBest:',NBest)
kf = LeaveOneOut()
for train_index,test_index in kf.split(train_y): # for each cv round
    train_cv = train_y.iloc[train_index]
    test_cv = train_y.iloc[test_index]
    test_sample = test_cv['sample_name'].values[0]
    print('Cross validation test sample:',test_sample)
    p_values_df = pd.read_pickle('../data/LOO_p_values/'+test_sample+'.pkl')
    kept_TCRs = p_values_df.T.nsmallest(NBest,'p_value').T.columns.values
    indices = [p_values_origin.columns.get_loc(col) for col in kept_TCRs]

    sub_train_X = train_X[:,indices].toarray()
    cv_train_X = sub_train_X[train_index,:]
    print('Shape of cv train data:',cv_train_X.shape)
    cv_test_X = sub_train_X[test_index,:]
    
    clf.fit(cv_train_X,train_cv['phenotype_status'])
    test_pred = clf.predict(cv_test_X)[0]
    test_proba = clf.predict_proba(cv_test_X)[:,1][0]
    y_true.append(test_cv['phenotype_status'].values[0])
    y_proba.append(test_proba)
    print('y_true:',test_cv['phenotype_status'].values[0],'y_pred:',test_pred,'y_proba:',test_proba)
    print()
        
print('loocv auroc: %.3f' % roc_auc_score(y_true,y_proba))
print('loocv log_loss: %.3f' % log_loss(y_true,y_proba))
            


