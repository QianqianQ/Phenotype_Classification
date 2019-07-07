# Import libraries

# Dataframe and arrays
import pandas as pd
import numpy as np

# Model
from MAP_estimator import MAP_estimator

# Helpers
import pickle
from datetime import datetime

from scipy.stats import fisher_exact,ttest_ind
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_validate,LeaveOneOut,KFold,StratifiedKFold,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,confusion_matrix,log_loss,make_scorer,classification_report

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# Warnings
import warnings
warnings.filterwarnings('ignore')


def cal_p_value1(present_c0,present_c1,class_counts,alternative='greater'):
    '''
    present_c0: the number of negative samples where the TCR present
    present_c1: the number of positive samples where the TCR present
    class_counts: the class distribution: [class_negative(0),class_positive(1)]
    alternative: Choose the type of test: greater, less, two_sided

    contingency table: [[present_c1, absent_c1], [present_c0, absent_c0]]
    '''

    absent_c0 = class_counts[0]-present_c0 # the number of negative samples where the TCR absent 
    absent_c1 = class_counts[1]-present_c1 # the number of positive samples where the TCR absent

    # Conducting fisher exact test 
    _, pvalue = fisher_exact([[present_c1, absent_c1], [present_c0, absent_c0]],alternative=alternative)
    return pvalue


def cal_p_value2(row,alternative='greater'):
    '''
    This function calculating Fisher exact test p-value based on pandas dataframe
    '''
    present_c0 = row['incidence in phenotype-']
    absent_c0 = row['absence in phenotype-']
    
    present_c1 = row['incidence in phenotype+']
    absent_c1 = row['absence in phenotype+']

    _, pvalue = fisher_exact([[present_c1, absent_c1], [present_c0, absent_c0]],alternative=alternative)
    return pvalue
    

def TCRs_selection_Fisher(train_neg,train_pos,TCRs,threshold,alternative='greater'):
    '''
    Select TCRs based on Fisher exact test p-values
    Args:
        train_neg: pandas dataframe, negative training data 
        train_pos: pandas dataframe, positive training data
        TCRs: a list of TCR candidates
        threshold: float(>0), p-value threshold 
        alternative: str, choosing the type of test: greater, less, two_sided

        The null hypothesis of the Fisher exact test is the TCR is not associated with the phenotype status
        alternative == 'greater': alternative hypothesis is TCR is associated with positive class
        alternative == 'less': alternative hypothesis is TCR is associated with negative class
    Return 
        a list of selected TCRs
    '''

    # sizes of samples in different classes
    N_c0 = train_neg.shape[0]
    N_c1 = train_pos.shape[0]

    '''
    Construct a TCR incidence dictionary
    key is TCR, value is [num_present_c0,num_present_c1]
    '''
    TCRs_inc = dict() # init dict
    for tcr in TCRs: # for each TCR in the list of TCR candidates

        neg_num = np.count_nonzero(train_neg[tcr].values) # the number of samples where the TCR occurs in the negative data
        pos_num = np.count_nonzero(train_pos[tcr].values) # the number of samples where the TCR occurs in the positive data

        if neg_num != 0 or pos_num != 0 : # if does not occur in any classes, not to add to the dict 
            TCRs_inc[tcr] = [neg_num,pos_num] # add the TCR incidence to dict

    # Compute fisher exact test for each TCR in the TCR incidence dict
    for tcr in TCRs_inc:
        present_c0, absent_c0, present_c1, absent_c1 = TCRs_inc[tcr][0], N_c0-TCRs_inc[tcr][0], TCRs_inc[tcr][1], N_c1-TCRs_inc[tcr][1]
        _, p_value = fisher_exact([[present_c1, absent_c1], [present_c0, absent_c0]],alternative=alternative)
        TCRs_inc[tcr].append(p_value) # store p-value

    inc_p_value_df = pd.DataFrame.from_dict(TCRs_inc,orient='index') # generate df from TCR incidence dict
    inc_p_value_df.columns = ['incidence in phenotype-','incidence in phenotype+','p_value'] # rename df

    TCRs_asso = inc_p_value_df[inc_p_value_df.p_value<=threshold].T.columns.values # select a list of pehnotype_associated TCRs
    return TCRs_asso


def TCRs_selection_ttest(train_neg_freq,train_pos_freq,TCRs,threshold):
    '''
    Select TCRs based on t-test p-values
    Args:
        train_neg_freq: pandas dataframe, negative training data (frequency version)
        train_pos_freq: pandas dataframe, positive training data (frequency version)
        TCRs:  a list of TCR candidates
        threshold: float(>0), p-value threshold 
    Return
        a list of selected TCRs
    '''

    def t_test(pos_arr,neg_arr):
        '''
        Compute t-test p-value of a TCR
        '''
        t_stat, pvalue = ttest_ind(pos_arr,neg_arr,equal_var=False)
        if t_stat > 0:
            pvalue = pvalue/2
        else:
            pvalue = 1-(pvalue/2)
        
        return pvalue

    TCRs_pvalue = dict() # init dict
    for tcr in TCRs: # for each TCR in the list of TCR candidates
        
        pos_arr = train_pos_freq[tcr] 
        neg_arr = train_neg_freq[tcr]

        if np.mean(pos_arr)!=0 or np.mean(neg_arr)!=0: # if does not occur in any classes, not to add to the dict 
            TCRs_pvalue[tcr] = [np.mean(pos_arr),np.mean(neg_arr)] # add the TCR incidence to dict
            p_value = t_test(pos_arr,neg_arr)
            TCRs_pvalue[tcr].append(p_value) # store p-value

    p_value_df = pd.DataFrame.from_dict(TCRs_pvalue,orient='index') # generate df from TCRs_pvalue dict
    p_value_df.columns = ['mean_positive','mean_negative','p_value'] # rename df

    TCRs_asso = p_value_df[p_value_df.p_value<=threshold].T.columns.values # select a list of pehnotype_associated TCRs
    
    return TCRs_asso


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

    
def parameter_search(clf, search_method, param_grid, train_X, train_y, test_X, test_y, nfolds=StratifiedKFold(n_splits=5, random_state=0), refit='AUC', n_iter=10):
    '''
    Perform grid search or random search to tune the parameters of the classifier
    '''

    print('Parameter_grid:',param_grid)

    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

    if search_method == 'grid':
        search = GridSearchCV(clf,param_grid,scoring=scoring,cv=nfolds,refit=refit)
    elif search_method == 'random':
        search = RandomizedSearchCV(clf,param_grid,n_iter=n_iter,scoring=scoring,cv=nfolds,refit=refit)
    else:
        raise ValueError('Invalid parameter search method!')

    search.fit(train_X, train_y)

    print("Best parameters set found on development set for %s:"%refit)
    print()
    print(search.best_params_)
    print()
    print('Best %s:'%refit, search.best_score_)
    print()
    print('Best estimator:')
    print(search.best_estimator_)
    print()
    
    print("Grid scores on training set:")
    print()
    means_roc = search.cv_results_['mean_test_AUC']
    stds_roc = search.cv_results_['std_test_AUC']

    means_acc = search.cv_results_['mean_test_Accuracy']
    stds_acc = search.cv_results_['std_test_Accuracy']
    for mean_roc, std_roc, mean_acc,std_acc, params in zip(means_roc, stds_roc, means_acc, stds_acc, search.cv_results_['params']):
        print("AUROC: %0.3f (+/-%0.03f), Accuracy: %0.3f (+/-%0.03f) for %r"
              % (mean_roc, std_roc * 2, mean_acc, std_acc*2, params))
    print()
    
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full testing set.")
    print()
    y_true, y_pred = test_y, search.predict(test_X)
    print(classification_report(y_true, y_pred))
    print('Accuracy: %.3f'%accuracy_score(y_true,y_pred))
    print()
    print('AUROC: %.3f'%roc_auc_score(y_true,search.predict_proba(test_X)[:,1]))
    print()
    print('Confusion matrix:')
    print(pd.DataFrame(confusion_matrix(y_true,y_pred),columns=['Predicted_negative','Predicted_positive'],
                    index=['Actual_negative','Actual_positive']))
    print()


def evaluate_param(clf, train_X, train_y, parameter, num_range, nfolds=StratifiedKFold(n_splits=5, random_state=0),ax=None):
    '''
    Evaluate the parameter of the classifier within the num_range, return a plot
    '''
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
    gs = GridSearchCV(clf, param_grid={parameter: num_range},cv=nfolds,scoring=scoring)
    gs.fit(train_X, train_y)
    results = gs.cv_results_
    X_axis = np.array(results['param_%s'%parameter].data)

    if ax is None:
        ax = plt.gca()

    for scorer, color in zip(sorted(scoring), ['g', 'k']):

        score_mean = results['mean_test_%s' % (scorer)]
        score_std = results['std_test_%s' % (scorer)]
        ax.fill_between(X_axis, score_mean - score_std,
                        score_mean + score_std,
                        alpha=0.1, color=color)
        ax.plot(X_axis, score_mean, color=color, alpha=1, label="%s" % (scorer))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    ax.set_xlabel(parameter)
    ax.set_ylabel("Score")
   
    return ax


def params_heatmap(clf, train_X, train_y, param_grid, scoring ='AUC', nfolds=StratifiedKFold(n_splits=5, random_state=0)):

    gs = GridSearchCV(clf, param_grid=param_grid,cv=nfolds,scoring=scoring)
    params_name = param_grid.keys()
    gs.fit(train_X, train_y)
    results = gs.cv_results_
    scores = np.array(results['mean_test_score'].reshape(len(param_grid[params_name[0]]),len(param_grid[params_name[1]])))
    scores_df = pd.DataFrame(scores,index=param_grid[params_name[0]],columns=param_grid[params_name[1]])
    sns.heatmap(scores_df, annot=True, vmin=0, vmax=1)


def rfecv(clf,train_X,train_y,test_X=None,test_y=None,cv=StratifiedKFold(n_splits=5, random_state=0),step=1,savefig=False,fig_name=None,return_selected=False):
    '''
    Perform  recursive feature elimination cross validation to select best number of features
    '''
    
    rfecv = RFECV(clf,cv=cv,scoring='roc_auc',step=step)
    rfecv.fit(train_X,train_y)
    
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    # plt.plot(np.linspace(1,(len(rfecv.grid_scores_)+1)*step,(len(rfecv.grid_scores_)+1)), rfecv.grid_scores_)
    plt.plot(range(1, (len(rfecv.grid_scores_)+1)), rfecv.grid_scores_)
    plt.xticks(reversed(range(len(rfecv.support_),0,-step)))
    if savefig == True:
        plt.savefig(fig_name)

    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Max cv auroc score: %.3f' % max(rfecv.grid_scores_))
    print()
    
    if test_X != None and test_y != None:
        print('Testing set:')
        print('AUROC: %.3f' % roc_auc_score(test_y,rfecv.predict_proba(test_X)[:,1]),
            ', Accuracy: %.3f' % accuracy_score(test_y,rfecv.predict(test_X)))
     
    if return_selected == True:
        supported = rfecv.get_support(indices=True)
        return supported

    
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