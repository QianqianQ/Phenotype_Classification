# Handle table-like data and matrices
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact,uniform
# Helpers
import os
import sys
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

# Posteriror probability of positive class
def predict_proba_c1(prior_c0,prior_c1,train_df,test_df):

    def pred_c1_novel(c0_counts,c1_counts, prior_c0, prior_c1, n, k):
        '''
        Compute positive-class posterior probability of a novel object
        Args:
            c0_counts - int(>=0) the number of negative samples in the training set
            c1_counts - int(>=0), the number of positive samples in the training set
            prior_c0 - [a_c0,b_c0], priors of negative class
            prior_c1 - [a_c1,b_c1], priors of positive class
            n - int(>=0), number of unique_TCRs of the novel object
            k - int(>=0), number of phenotype_associated_TCRs of the novel object
        Return:
            positive-class posterior probability 
        '''
        N = [c0_counts, c1_counts]

        # log-posterior odds ratio
        F = log(N[1] + 1) - log(N[0] + 1) + betaln(prior_c0[0], prior_c0[1]) - betaln(prior_c1[0],prior_c1[1]) + \
            betaln(k + prior_c1[0], n - k + prior_c1[1]) - betaln(k + prior_c0[0], n - k + prior_c0[1])

        post_prob_c1 = exp(F)/(1+exp(F)) # positive-class posterior probability
        return post_prob_c1

    # compute each sample in the df
    c0_counts = train_df[train_df['phenotype_status']==0].shape[0]
    c1_counts = train_df[train_df['phenotype_status']==1].shape[0]
    # get test_sample name
    test_sample = test_df['sample_name'].tolist()[0]
    pred__prob_c1 = [pred_c1_novel(c0_counts,c1_counts,prior_c0,prior_c1,row['unique_TCRs'],row[test_sample])
     for _,row in test_df.iterrows()]

    return pred__prob_c1

# Inner cv of the nest cv
def LOOCV_inner(prior_c0,prior_c1,train):
    '''
    The inner cross validation calculates a LOOCV AUROC under specific priors
    '''
    y_true = []
    y_proba = []
    
    kf = LeaveOneOut()
    for train_index,test_index in kf.split(train):
        train_cv, test_cv = train.iloc[train_index], train.iloc[test_index] # get training samples and one testing sample
        pred_prob = predict_proba_c1(prior_c0,prior_c1,train_cv,test_cv)[0]
        y_true.append(test_cv['phenotype_status'].tolist()[0])
        y_proba.append(pred_prob)
    cv_auc = roc_auc_score(y_true,y_proba)
    return cv_auc

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

    a0min, a0max, a0step = 1, 5, 0.5
    b0min, b0max, b0step = 18000, 22000,1000
    a0 = np.arange(a0min, a0max + a0step, a0step)
    b0 = np.arange(b0min, b0max + b0step, b0step)
    # a0, b0 = np.meshgrid(np.arange(a0min, a0max + a0step, a0step), np.arange(b0min, b0max + b0step, b0step))

    a1min, a1max, a1step = 8, 20, 1
    b1min, b1max, b1step = 10000, 13000,1000
    a1 = np.arange(a1min, a1max + a1step, a1step)
    b1 = np.arange(b1min, b1max + b1step, b1step)
    # a1, b1 = np.meshgrid(np.arange(a1min, a1max + a1step, a1step), np.arange(b1min, b1max + b1step, b1step))

    print(n)
    flag = 0
    kf = LeaveOneOut()
    for train_index,test_index in kf.split(train_origin): # for each cv round
        if flag == n:
            break
        else:
            flag += 1

    print('test index:',test_index)
    train = train_origin.copy(deep=True) # a copy of the original training data
    train_cv, test_cv = train.iloc[train_index], train.iloc[test_index] # get training samples and one testing sample
    # Select a list of associated TCRs based on count df of training samples and threshold
    count_train = count_df[count_df['sample_name'].isin(train_cv['sample_name'])] # count df of training samples
    count_test = count_df[count_df['sample_name'].isin(test_cv['sample_name'])] # count df of the testing sample

    TCRs_asso = TCRs_selection(count_train,TCRs,0.2) # select a list of associated TCRs

    test_sample = test_cv['sample_name'].tolist()
    test_name = test_sample[0]
    # training set
    train_sample = train_cv['sample_name'].tolist()
    train_asso = []
    for i in range(len(train_sample)): # for each training sample

        temp_train = count_train.loc[count_train.sample_name==train_sample[i]] # count df of the training sample
        i_asso = np.count_nonzero(temp_train[TCRs_asso].values) # count the number of phenotype_associated TCRs in this sample
        train_asso.append(i_asso)

    train_cv[test_name] = train_asso # add the 'phenotype_associated_TCRs' column to the training data


    # testing set, the same steps as the above

    test_asso = []
    for i in range(len(test_sample)): # for each testing sample (in LOOCV, only one)

        temp_test = count_test.loc[count_test.sample_name==test_sample[i]]
        i_asso = np.count_nonzero(temp_test[TCRs_asso].values)
        test_asso.append(i_asso)

    test_cv[test_name] = test_asso

    # Select hyperparamers
    test_sample = test_cv['sample_name'].tolist()[0]
    train_cv2 = pd.read_csv('../data/CV/'+test_sample+'.csv')

    best_auc = -np.Inf
    best_priors = None
    for priors in product(a0,b0,a1,b1):# nested for-loops
        prior_c0 = [priors[0],priors[1]]
        prior_c1 = [priors[2],priors[3]]
        cv_auc = LOOCV_inner(prior_c0,prior_c1,train_cv2) # conduct inner cv for this set of priors 
        if cv_auc > best_auc:
            best_priors = priors
            best_auc = cv_auc

    # the hyperparamer is chosen, predict on the testing set
    prior_c0 = [best_priors[0],best_priors[1]]
    prior_c1 = [best_priors[2],best_priors[3]]
    test_proba = predict_proba_c1(prior_c0,prior_c1,train_cv,test_cv)[0]

    print('test sample: ',test_cv['sample_name'].tolist()[0], ' unique_TCRs: ',test_cv['unique_TCRs'].tolist()[0],' associated_TCRs: ',i_asso)        
    print('y_true:',test_cv['phenotype_status'].tolist()[0],' y_proba_c1: %.3f'%test_proba)
    print()

res = pd.DataFrame({'sample_name':[test_cv['sample_name'].tolist()[0]],
                    'phenotype_status':[test_cv['phenotype_status'].tolist()[0]],'pred_proba':[test_proba]})

res.to_csv('outputs/CV2/'+test_sample+'.csv',index=False)


