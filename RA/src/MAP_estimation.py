import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0,'../../')
from argparse import ArgumentParser

from MAP_estimator import MAP_estimator

from helpers import *
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score,log_loss

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

import time

import warnings
warnings.filterwarnings('ignore')
    

if __name__=='__main__':

    parser = ArgumentParser()
    # parser.add_argument('-i','--input',nargs=2,dest='input',type=str,
    #                     help='input files: arg1: TCRs stats csv file,\
    #                                     arg2: count df pkl file')
    parser.add_argument('-n','--num',dest='num',type=int,
                        help='the nth sample left out as test sample')
    parser.add_argument('-t','--threshold',dest='threshold',type=float,help='p-value threshold')
    parser.add_argument('-s','--store',dest='store',type=int,
                        help='store in the folder s')

    # Parsing of parameters
    try:
        args = parser.parse_args()
    except:
        # parser.print_help()
        sys.exit(1)

    try:
        threshold = args.threshold
        n = args.num
        s = args.store
    except:
        raise ValueError('Wrong input!')

    #-------------------------------------------------------------------------------------------------------------------

    vgam=importr('VGAM')

    robjects.r('''

        betabin_fit<- function(n,k){
        n<- as.numeric(n)
        k<-as.numeric(k)
        fit <- vglm(cbind(k, n - k) ~ 1, betabinomialff,trace=TRUE,imethod=4)
        Coef(fit)
        }
    ''')

    fit_model = robjects.globalenv['betabin_fit']

    def prior_init(train):
        # subdataframes of different classes
        train_c0 = train[train['phenotype_status']==0]
        train_c1 = train[train['phenotype_status']==1]
        
        # lists n, k, k/n of the negative class
        n_c0 = train_c0['unique_TCRs'].tolist()
        k_c0 = train_c0['phenotype_associated_TCRs'].tolist()
        
        # lists n, k, k/n of the positive class
        n_c1 = train_c1['unique_TCRs'].tolist()
        k_c1 = train_c1['phenotype_associated_TCRs'].tolist()
        
        coef0 = np.array(fit_model(n_c0,k_c0)).tolist()
        coef1 = np.array(fit_model(n_c1,k_c1)).tolist()
        
        return [coef0,coef1]

    def MAP_predict(train,test,prior_c0,prior_c1):
        '''
        Predicting testing data
        '''
        MAP = MAP_estimator(prior_c0,prior_c1) # construct a MAP_estimator instance
        MAP.fit(train,'unique_TCRs','phenotype_associated_TCRs','phenotype_status') # train the model using training set
        print('optimized priors:', 'class 0:', list(np.around(MAP.priors()[0],3)),
            ', class 1:', list(np.around(MAP.priors()[1],3))) # print the optimized priors

        y_pred = MAP.predict(test,'unique_TCRs','phenotype_associated_TCRs')[0] # predict the label
        y_proba = MAP.predict_proba_pos(test,'unique_TCRs','phenotype_associated_TCRs')[0] # compute the positive-class posterior probability

        return y_pred, y_proba

    print('The %dth samples as the test sample'%n)
    data_path = '../data/'
    data= pd.read_csv(data_path+'MAP_data.csv')
    count_df = pd.read_pickle(data_path+'count_df.pkl')
    label = pd.read_csv(data_path+'train_y.csv')
    count_df = pd.merge(label, count_df, on='sample_name')
    data = pd.merge(label, data, on='sample_name')

    TCRs = count_df.drop(['sample_name','phenotype_status'],axis=1).columns.values

    flag = 0
    kf = LeaveOneOut()
    for train_index,test_index in kf.split(data): # for each cv round
        if flag == n:
            break
        else:
            flag += 1

    train = data.copy(deep=True) # a copy of the original training data
    train_cv, test_cv = train.iloc[train_index], train.iloc[test_index] # get training samples and one testing sample

    # Select a list of associated TCRs based on count df of training samples and threshold
    count_train = count_df[count_df['sample_name'].isin(train_cv['sample_name'])] # count df of training samples
    count_test = count_df[count_df['sample_name'].isin(test_cv['sample_name'])] # count df of the testing sample

    try:
        count_train_neg = count_train[count_train['phenotype_status']==0]
        count_train_pos = count_train[count_train['phenotype_status']==1]
    except KeyError:
        neg_samples = train_cv[train_cv['phenotype_status']==0]['sample_name']
        pos_samples = train_cv[train_cv['phenotype_status']==1]['sample_name']
        count_train_neg = count_train[count_train['sample_name'].isin(neg_samples)]
        count_train_pos = count_train[count_train['sample_name'].isin(pos_samples)]

    TCRs_asso = TCRs_selection_Fisher(count_train_neg,count_train_pos,TCRs,threshold) # select a list of TCRs

    '''
    Get statistics: number of phenotype_associated_TCRs of each sample
    '''
    # training set
    train_sample = train_cv['sample_name'].tolist()
    train_asso = []
    for i in range(len(train_sample)): # for each training sample

        temp_train = count_train.loc[count_train.sample_name==train_sample[i]] # count df of the training sample
        i_asso = np.count_nonzero(temp_train[TCRs_asso].values) # count the number of phenotype_associated TCRs in this sample
        train_asso.append(i_asso)

    train_cv['phenotype_associated_TCRs'] = train_asso # add the 'phenotype_associated_TCRs' column to the training data


    # testing set, the same steps as the above
    test_sample = test_cv['sample_name'].tolist()
    test_asso = []
    for i in range(len(test_sample)): # for each testing sample (in LOOCV, only one)

        temp_test = count_test.loc[count_test.sample_name==test_sample[i]]
        i_asso = np.count_nonzero(temp_test[TCRs_asso].values)
        test_asso.append(i_asso)

    test_cv['phenotype_associated_TCRs'] = test_asso


    priors_init = prior_init(train_cv)
    print('priors initialization:','class 0:', list(np.around(np.array(priors_init[0]),3)),
        ', class 1:', list(np.around(np.array(priors_init[1]),3)))
    prior_c0 = priors_init[0]
    prior_c1 = priors_init[1]
    test_pred, test_proba = MAP_predict(train_cv,test_cv,prior_c0,prior_c1)

    print('test sample:',test_sample, ', unique_TCRs:',test_cv['unique_TCRs'].tolist()[0],', associated_TCRs:',test_cv['phenotype_associated_TCRs'].tolist()[0])           
    print('y_true:',test_cv['phenotype_status'].tolist()[0],', y_pred:',test_pred,
        ', y_proba_pos: %.3f'%test_proba)

    test_sample = test_cv['sample_name'].tolist()[0]
    df = pd.DataFrame({'sample_name':[test_sample],'y_true':[test_cv['phenotype_status'].values[0]],
                        'y_proba':[test_proba]})
    df.to_pickle('LOO_CV_MAP_'+str(s)+'/'+test_sample+'.pkl')
