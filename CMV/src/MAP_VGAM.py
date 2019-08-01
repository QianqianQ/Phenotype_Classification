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
    parser.add_argument('-t','--threshold',dest='threshold',type=float,help='p-value threshold')
    parser.add_argument('-s','--store',dest='store',type=int,
                        help='store in the folder s')
    parser.add_argument('-n','--num',dest='num',type=int,
                        help='the nth sample left out as test sample')
 

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

    # utils = importr('utils')
    # utils.install_packages('VGAM',repos = "http://cran.us.r-project.org")
    vgam=importr('VGAM')

    robjects.r('''

        betabin_fit<- function(n,k){
        n<- as.numeric(n)
        k<-as.numeric(k)
        fit <- vglm(cbind(k, n - k) ~ 1, betabinomialff,trace=TRUE)
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
    train_y = pd.read_csv('../data/train_Y.csv')
    flag = 0
    kf = LeaveOneOut()
    for train_index,test_index in kf.split(train_y): # for each cv round
        if flag == n:
            break
        else:
            flag += 1

    train_cv, test_cv = train_y.iloc[train_index], train_y.iloc[test_index]
    test_sample = test_cv['sample_name'].values[0]
    print('Cross validation test sample:',test_sample)

    MAP_data = pd.read_pickle('../data/LOO_MAP_data/'+test_sample+'.pkl')

    MAP_data['phenotype_associated_TCRs']= MAP_data[threshold]
    train_MAP = MAP_data[MAP_data.sample_name.isin(train_cv['sample_name'])]
    test_MAP = MAP_data[MAP_data.sample_name.isin(test_cv['sample_name'])]
    print(train_MAP.shape,test_MAP.shape)

    priors_init = prior_init(train_MAP)
    print('priors initialization:','class 0:', list(np.around(np.array(priors_init[0]),3)),
        ', class 1:', list(np.around(np.array(priors_init[1]),3)))
    prior_c0 = priors_init[0]
    prior_c1 = priors_init[1]
    test_pred, test_proba = MAP_predict(train_MAP,test_MAP,prior_c0,prior_c1)

    print('test sample:',test_sample, ', unique_TCRs:',test_MAP['unique_TCRs'].tolist()[0],', associated_TCRs:',test_MAP['phenotype_associated_TCRs'].tolist()[0])           
    print('y_true:',test_MAP['phenotype_status'].tolist()[0],', y_pred:',test_pred,
        ', y_proba_pos: %.3f'%test_proba)
    
    df = pd.DataFrame({'sample_name':[test_sample],'y_true':[test_cv['phenotype_status'].values[0]],
                        'y_proba':[test_proba]})
    df.to_pickle('LOO_CV_MAP_'+str(s)+'/'+test_sample+'.pkl')
  