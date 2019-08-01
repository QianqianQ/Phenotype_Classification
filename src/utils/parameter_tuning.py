# Import libraries

# Dataframe and arrays
import pandas as pd
import numpy as np

# Helpers
import pickle
from datetime import datetime

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