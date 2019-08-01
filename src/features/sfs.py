#!/usr/bin/env python
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=20G
#SBATCH -o ssfs%J.out

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Helpers
import os
import sys

# Prediction
from math import log
from scipy.special import beta, comb
from math import exp


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score,confusion_matrix
from sklearn.model_selection import LeaveOneOut,KFold,StratifiedKFold


from sklearn.ensemble import RandomForestClassifier

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

if __name__=='__main__':


    train = pd.read_pickle('../data/feature_engineering/sub_train_bin.pkl')
    test = pd.read_pickle('../data/feature_engineering/sub_test_bin.pkl')
    
    train_X = train.drop(['sample_name','phenotype_status'],axis=1).values
    test_X = test.drop(['sample_name','phenotype_status'],axis=1).values
    
    train_y = train['phenotype_status'].values
    test_y = test['phenotype_status'].values

    # Define classifier
    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                  max_depth=31, max_features='auto', max_leaf_nodes=None,
                                  min_impurity_decrease=0.0, min_impurity_split=None,
                                  min_samples_leaf=1, min_samples_split=5,
                                  min_weight_fraction_leaf=0.0, n_estimators=180, n_jobs=1,
                                  oob_score=True, random_state=0, verbose=0, warm_start=False)

    sfs = SFS(clf,
              k_features=(70,110),
              forward = True,
              floating= False,
              verbose=2,
              scoring='roc_auc',
              cv=5)

    sfs = sfs.fit(train_X,train_y)

    print('CV Score:')
    print(sfs.k_score_)
    print(sfs.k_feature_idx_)
    #-------------------------------------------------------------------------------------------------------------------
