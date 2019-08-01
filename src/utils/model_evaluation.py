# Import libraries

# Dataframe and arrays
import pandas as pd
import numpy as np

# Model
import sys
sys.path.insert(0,'../models/')
from MAP_estimator import MAP_estimator

# Helpers
import pickle
from datetime import datetime

from sklearn.model_selection import cross_validate,LeaveOneOut,KFold,StratifiedKFold
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,confusion_matrix,log_loss,make_scorer,classification_report

# Warnings
import warnings
warnings.filterwarnings('ignore')


def LOOCV_MAP(train_origin,count_df,TCRs,threshold,priors_init_value=None,prior_init_fun=None,verbose=True,output=False, output_file=None):
    '''
    Perform Leave-One-Out cross validation MAP estimation. The list of phenotype_associated_TCRs is selected by Fisher exact test. 
    Args:
        train_origin: training dataframe, containing the columns 'sample_name', 'unique_TCRs' and 'phenotype_status' 
        count_df: count version dataframe
        TCRs: a list of TCR candidates
        threshold: p-value threshold
        priors_init_value: [[a_c0,b_c0],[a_c1,b_c1]], beta prior initialization values
        prior_init_fun: the function used to initialize priors
        verbose: bool value, whether to output more information
        output: bool, whether to save the result dataframe as a output file
        output_file: str, path+name of the output file
    '''
    def MAP_predict(train,test,prior_c0,prior_c1):
        '''
        Predicting testing data
        '''
        MAP = MAP_estimator(prior_c0,prior_c1) # construct a MAP_estimator instance
        MAP.fit(train,'unique_TCRs','phenotype_associated_TCRs','phenotype_status') # train the model using training set
        if verbose == True:
            print('optimized priors:', 'class 0:', list(np.around(MAP.priors()[0],3)),
                ', class 1:', list(np.around(MAP.priors()[1],3))) # print the optimized priors

        y_pred = MAP.predict(test,'unique_TCRs','phenotype_associated_TCRs') # predict the label
        y_proba = MAP.predict_proba_pos(test,'unique_TCRs','phenotype_associated_TCRs') # compute the positive-class posterior probability
        
        return y_pred, y_proba

    # Init lists
    sample_name = []
    y_true = []
    y_pred = []
    y_proba = [] 
    round_count = 0 # count the rounds of cross validation
    LOO = LeaveOneOut()
    for train_index,test_index in LOO.split(train_origin): # for each cv round
        round_count += 1
        print('Round',round_count)

        train = train_origin.copy(deep=True) # a copy of the original training data
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

        '''
        Train the estimator, predict testing set (testing sample)
        '''
        if priors_init_value is None and prior_init_fun is None:
            raise ValueError('No prior initialization value or method provided')

        elif priors_init_value is not None and prior_init_fun is None: # a set of uniform prior values for all the cv rounds
            prior_c0 = priors_init_value[0] 
            prior_c1 = priors_init_value[1]
            test_pred, test_proba = MAP_predict(train_cv,test_cv,prior_c0,prior_c1)

        elif priors_init_value is None and prior_init_fun is not None: # perform a prior initialization method
            priors_init = prior_init_fun(train_cv)
            if verbose == True:
                print('priors initialization:','class 0:', list(np.around(np.array(priors_init[0]),3)),
                    ', class 1:', list(np.around(np.array(priors_init[1]),3)))
            prior_c0 = priors_init[0]
            prior_c1 = priors_init[1]
            test_pred, test_proba = MAP_predict(train_cv,test_cv,prior_c0,prior_c1)
        else:
            priors_init = prior_init_fun(train_cv,priors_init_value)
            if verbose == True:
                print('priors initialization:','class 0:', list(np.around(np.array(priors_init[0]),3)),
                    ', class 1:', list(np.around(np.array(priors_init[1]),3)))
            prior_c0 = priors_init[0]
            prior_c1 = priors_init[1]
            test_pred, test_proba = MAP_predict(train_cv,test_cv,prior_c0,prior_c1)

        # append results to lists, round to 3 decimal points
        sample_name.append(test_cv['sample_name'].tolist()[0])
        y_true.append(test_cv['phenotype_status'].tolist()[0])
        y_pred.append(test_pred[0])
        y_proba.append(test_proba[0])
        
        # Results of this round
        if verbose == True:
            print('the number of associated TCRs in this round:',len(TCRs_asso))
            print('test sample:',test_cv['sample_name'].tolist()[0], ', unique_TCRs:',test_cv['unique_TCRs'].tolist()[0],', associated_TCRs:',i_asso)        
            print('Result:','y_true:',test_cv['phenotype_status'].tolist()[0],', y_pred:',test_pred[0],
                ', y_proba_pos: %.3f'%test_proba[0])
            print()

    print('loocv auroc: %.3f' % roc_auc_score(y_true,y_proba))
    print('loocv log_loss: %.3f' % log_loss(y_true,y_proba))
    print()

    # Save the prediction as df
    prediction = pd.DataFrame({'sample_name':sample_name,'y_true':y_true,'y_pred':y_pred,'y_proba_pos':y_proba})

    if output == True: # Save the prediction df as csv file
        
        if output_file is not None:
            prediction.to_csv(output_file, index=False)
        else:
            now = datetime.now()
            currentDate = str(now.day) + "-" + str(now.month) + "-" + str(now.year)+"-"+str(now.hour)+"-"+str(now.minute)+"-"+str(now.second)
            prediction.to_csv('MAP_pred_'+currentDate+'.csv',index=False)
        
    return roc_auc_score(y_true,y_proba), log_loss(y_true,y_proba), prediction

    
def estimator_result(clf, train_X, train_y, test_X, test_y, n_folds=StratifiedKFold(n_splits=5, random_state=0), printFeatureImportance=False, features=None):
    '''
    Perform cross validation and prediction on training set (and testing set) 
    '''
    
    print(clf)
    
    # cross validation
    print("\nCross validation:")
    cv_results = cross_validate(clf,train_X,train_y,scoring=('accuracy', 'roc_auc'),cv=n_folds)
    print('accuracy score: %.3f' % np.mean(cv_results['test_accuracy']))
    print('AUROC: %.3f' % np.mean(cv_results['test_roc_auc']))

    # fit training set
    clf.fit(X=train_X, y=train_y)
    
    #Print Feature Importance:
    if printFeatureImportance and features != None:
        feat_imp = pd.Series(clf.feature_importances_,features).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

    #----------------------------------------------------------------------------------------------------------------------------
    print("_" * 80)
    print("Training set:")
    # predict train_y
    predict_train = clf.predict(train_X)
    print('accuracy score: %.3f' % accuracy_score(train_y, predict_train))
    
    # AUROC of training set
    predict_train_prob = clf.predict_proba(train_X)
    pos_prob_train = predict_train_prob[:, 1]
    print('AUROC: %.3f' % roc_auc_score(train_y, pos_prob_train))

    # log loss of training set
    print('log-loss: %.3f' % log_loss(train_y, predict_train_prob))

    #---------------------------------------------------------------------------------------------------------------------------
    print("_" * 80)
    print("Testing set;")
    # predict test_Y
    predict_test = clf.predict(test_X)
    print('accuracy score: %.3f' % accuracy_score(test_y, predict_test))

    # AUROC of testing set
    predict_test_prob = clf.predict_proba(test_X)
    pos_prob_test = predict_test_prob[:, 1]
    print('AUROC: %.3f' % roc_auc_score(test_y, pos_prob_test))

    # log loss of testing set
    print('log-loss: %.3f' % log_loss(test_y, predict_test_prob))

    print('classification_report')
    print(classification_report(test_y, predict_test))

    print('Confusion matrix:')
    print(pd.DataFrame(confusion_matrix(test_y, predict_test), columns=['Predicted_negative','Predicted_positive'],
                        index=['Actual_negative','Actual_positive']))

    print('*'*80)