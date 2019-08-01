# Handle table-like data and matrices
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact,uniform
# Helpers
import os
import sys
from argparse import ArgumentParser
sys.path.insert(0,'../')
from scipy.special import digamma,betaln
import time
from scipy.optimize import minimize
from itertools import product
# Prediction
from helpers import *

from numpy import random
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score,confusion_matrix,log_loss
from sklearn.model_selection import LeaveOneOut,KFold,StratifiedKFold

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D

# import plotly
# plotly.tools.set_credentials_file(username='tracyqin326', api_key='EICCf5vuIzI5hVfA4gYC')
# import plotly.plotly as py
# import plotly.graph_objs as go

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

    train_origin = pd.read_csv('../data/'+'train.csv')
    count_df = pd.read_pickle('../data/'+'count_df.pkl')
    TCRs = count_df.drop(['sample_name','phenotype_status'],axis=1).columns.values

    print(n)
    flag = 0
    kf = LeaveOneOut()
    for train_index,test_index in kf.split(train_origin): # for each cv round
        print(train_index,test_index)
        if flag == n:
            break
        else:
            flag += 1

    train = train_origin.copy(deep=True) # a copy of the original training data
    train_cv, test_cv = train.iloc[train_index], train.iloc[test_index] # get training samples and one testing sample

    test_sample = test_cv['sample_name'].tolist()[0]
    print(test_sample)
    # Select a list of associated TCRs based on count df of training samples and threshold
    count_train = count_df[count_df['sample_name'].isin(train_cv['sample_name'])] # count df of training samples
    count_test = count_df[count_df['sample_name'].isin(test_cv['sample_name'])] # count df of the testing sample

    kf2 = LeaveOneOut()
    for train_index2,test_index2 in kf2.split(train_cv):
        train_cv2, test_cv2 = train_cv.iloc[train_index2], train_cv.iloc[test_index2] # get training samples and one testing sample

        # Select a list of associated TCRs based on count df of training samples and threshold
        count_train2 = count_train[count_train['sample_name'].isin(train_cv2['sample_name'])] # count df of training samples
        count_test2 = count_train[count_train['sample_name'].isin(test_cv2['sample_name'])] # count df of the testing sample

        TCRs_asso = TCRs_selection(count_train2,TCRs,0.2) # select a list of associated TCRs

        train_cv[test_cv2['sample_name'].tolist()[0]] = np.count_nonzero(count_train[TCRs_asso],axis=1)

    train_cv.to_csv('../data/CV/'+test_sample+'.csv',index=False)

