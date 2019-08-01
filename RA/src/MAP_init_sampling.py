# Import libraries

# dataframe and arrays
import pandas as pd
import numpy as np

# Model
from classifiers import MAP_estimator, cal_p_value, TCRs_selection, LOOCV_MAP

# Helpers
import pickle
from datetime import datetime
from argparse import ArgumentParser

from numpy.random import uniform
from sklearn.model_selection import LeaveOneOut,KFold,StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score,confusion_matrix, log_loss

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

import os
import sys
import warnings
warnings.filterwarnings('ignore')


if __name__=='__main__':

    '''
    Command line arguments:
        -i arg1 arg2: input file names, arg1 - TCRs stats csv file, containing number of unique_TCRs of each sample, arg2 - count/binary version data
        -t arg: p-value threshold
        -p0 arg1 arg2: beta prior initialization of negative class, arg1 - a_c0, arg2 - b_c0
        -p1 arg1 arg2: beta prior initialization of positive class, arg1 - a_c1, arg2 - b_c1
        -m arg: optimization method
        -r arg: the number of rounds sampling new priors and then computing LOOCV MAP
    '''
    parser = ArgumentParser()
    parser.add_argument('-i','--input',nargs=2,dest='input',type=str,
                        help='input files: arg1: TCRs stats csv file,\
                                        arg2: count/binary df pkl file')
    parser.add_argument('-t','--threshold',dest='threshold',type=float,help='p-value threshold')
    parser.add_argument('-p0','--priors0',dest='priors_c0',type=float,nargs=2,
                        help='priors_c0 initialization')
    parser.add_argument('-p1','--priors1',dest='priors_c1',type=float,nargs=2,
                        help='priors_c0 initialization')
    parser.add_argument('-m','--method',dest='opt_method',default='L-BFGS-B',type=str,help='optimization method')
    parser.add_argument('-r','--rounds',dest='rounds',default=20,type=int,help='')

    # Parsing of parameters
    try:
        args = parser.parse_args()
    except:
        # parser.print_help()
        sys.exit(1)

    try:
        # TCRs stats file
        TCRs_stats_file = args.input[0]
        if os.path.isfile(TCRs_stats_file) == False:
            raise FileNotFoundError('TCR stats file not found!')
        train_origin = pd.read_csv(TCRs_stats_file)

        # count df
        count_df_file = args.input[1]
        if os.path.isfile(count_df_file) == False:
            raise FileExistsError('Count df file not found!')
        count_df = pd.read_pickle(count_df_file)

        # threshold 
        threshold = args.threshold
        # priors_c0 initialization
        prior_c0_init = args.priors_c0
        # priors_c1 initialization
        prior_c1_init = args.priors_c1
        # optimization method
        opt_method = args.opt_method
        # rounds of computation
        rounds = args.rounds

    except:
        raise ValueError('Wrong input!')

    #-------------------------------------------------------------------------------------------------------------------

    print('prior_c0 initialization:',prior_c0_init,'prior_c1 initialization:',prior_c1_init)
    print('p-value threshold:',threshold)
    print('Sampling from uniform distribution\n')

    priors_init = [prior_c0_init,prior_c1_init]

    priors = []
    auroc = []
    log_loss = []

    count = 0
    while count < rounds: # run 20 rounds

        # New priors sampling from uniform distribution
        a_c0 = uniform(priors_init[0][0]-1,priors_init[0][0]+1,1)
        b_c0 = uniform(priors_init[0][1]-1000,priors_init[0][1]+1000,1)

        a_c1 = uniform(priors_init[1][0]-1,priors_init[1][0]+1,1)
        b_c1 = uniform(priors_init[1][1]-1000,priors_init[1][1]+1000,1)

        new_init = [[a_c0,b_c0],[a_c1,b_c1]] # new prior initialization
        print('new prior initialization:',new_init)

        # Running LOOCV MAP 
        auroc_i, log_loss_i = LOOCV_MAP(train_origin,count_df,threshold,priors_init_value=new_init,verbose=False)
        print()

        # Store results
        priors.append(new_init)
        auroc.append(auroc_i)
        log_loss.append(log_loss_i)

        count += 1
    
    best_auroc = max(auroc) # best auroc
    best_priors = priors[auroc.index(best_auroc)] # best priors 
    print('The init:', best_priors, 'has best auroc:%.3f'%best_auroc,'with log loss %.3f'%log_loss[auroc.index(best_auroc)])
