"""
This object is a MAP (maximum a posteriori) estimator for predicting serostatuses based on TCR repertoire. 
The idea is from the paper:
"Immunosequencing identifies signatures of cytomegalovirus exposure history and HLA-mediated effects on the T cell repertoire"(Emerson et al., 2017)

The input data format is pandas dataframes with columns 'phenotype_status'(target), 'unique_TCRs' and 'phenotype_associated_TCRs' (Could be customed by users)
Methods include:
fit: Train the MAP model based on the training data
predict: Predict class labels of the testing data
predict_prob_pos: Estimate posteriror probability for positive class of the testing data
priors: Get priors for this estimator
"""
#==============================================================================================================================================================
__author__ = 'Qin Qianqian'
__email__ = 'qianqian.qin@outlook.com'
__status__ = 'Development'


# Libraries
# Dataframe and arrays
import pandas as pd
import numpy as np

# Functions used in the model
import math
from math import log, exp, isnan
from scipy.special import beta,comb,digamma,betaln
from scipy.optimize import minimize

# Visualisation
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Warnings and exceptions
import warnings
warnings.filterwarnings('ignore')
from sklearn.exceptions import NotFittedError


class MAP_estimator:
    def __init__(self,prior_c0,prior_c1,opt_method='L-BFGS-B',jac=True,learning_rate=1e-2,precision=1e-6,max_iter=50000,obj_plot=False):
        '''
        prior_c0: list [a_c0,b_c0], beta prior initailization for class 0 (negative class)
        prior_c1: list [a_c1,b_c1], beta prior initailization for class 1 (positive class)
        opt_method: str, method used for optimize beta priors
        jac: bool value, whether to use jacobian function in the process of optimization

        learning_rate: float (>0), learning_rate of optimization
        precision: float (>0), precision threshold of optimization
        max_iter: int (>0), maximum number of iterations in optimization

        obj_plot: bool value, whether to plot curve that objective function value changes with number of iterations

        NB: in current version of estimator, 3 properties: learning_rate, precision and max_iter, are not used
        '''
        self.prior_c0 = prior_c0
        self.prior_c1 = prior_c1

        self.opt_method = opt_method
        self.jac = jac
        self.lr = learning_rate
        self.precision = precision
        self.max_iter = max_iter
        self.obj_plot = obj_plot

        '''
        total_counts: the total number of training samples,initializing as None before training the model 
        c0_counts: the number of negative training samples, initializing as None before training the model 
        c1_counts: the number of positive training samples, initializing as None before training the model 
        '''
        self.total_counts = None
        self.c0_counts = None
        self.c1_counts = None

    
    def fit(self,df,uniq_TCRs,phenotype_asso_TCRs,phenotype_status):
        '''
        Train the model

        Arg: 
            df - pandas dataframe of training data
            uniq_TCRs - str, the column name representing the number of unique TCRs
            phenotype_asso_TCRs - str, the column name representing the number of TCRs associated with the phenotype status
            phenotype_status - str, the column name representing the phenotype_status (label)

        3 functions: 
            neg_objective: Computing negative objective function
            neg_obj_jac: Computing jacobian of negtive objective
            optimize_prior: Optimizing priors by minimizing neg_objective
        '''

        def neg_objective(priors,n,k):
            '''
            Compute negative objective function value
            Args:
                priors - list [a,b], beta prior
                n - list of the number of unique_TCRs
                k - list of the number of phenotype_associated_TCRs

                NB: n, k are lists of specific class (negative/positive)
            '''
            a = priors[0] # parameter a of beta 
            b = priors[1] # parameter b of beta
            N_l = len(n) # number of samples 
            
            '''
            Compute objective function value
            '''
            sum_log_beta = 0
            for i in range(N_l):
                sum_log_beta += betaln(k[i]+a,n[i]-k[i]+b)

            obj = -N_l*betaln(a,b)+sum_log_beta

            obj_value.append(obj) # store objective function value, for the purpose of plotting 

            return -obj # return negative objective value
        
        def neg_obj_jac(priors,n,k):
            '''
            Compute jacobian matrix of negative objective function with respect to a and b.
            The same args as neg_objective function
            '''
            a = priors[0]
            b = priors[1]
            N_l = len(n)

            '''
            Compute jacobian matrix
            '''
            sum_a = 0
            sum_b = 0
            for i in range(N_l):
                sum_a += digamma(k[i] + a) - digamma(n[i] + k[i] + a + b)
                sum_b += digamma(n[i] - k[i] + b) - digamma(n[i] + k[i] + a + b)

            gradient_a = -(-N_l * (digamma(a) - digamma(a + b)) + sum_a)
            gradient_b = -(-N_l * (digamma(b) - digamma(a + b)) + sum_b)

            return np.array((gradient_a,gradient_b))

        def optimize_prior(sub_df,priors,opt_method, jac, jacobian_fun, learning_rate, precision, max_iterations):   
            '''
            Optimizing beta prior by minimizing negative objective function (maximizing the joint likelihood on the training set)
            Args:
                sub_df - sub dataframe of the class to optimize
                priors - [a,b], prior initialization
                opt_method - str, optimization method
                jac - bool value, whether to use jacobian in the process of optimization
                jacobian_fun - 1*2 array, jacobian matrix
                learning_rate, precision, max_iterations - parameters related to optimization, not used in current code
            Return:
                array of optimized priors
            '''
            
            n = sub_df[uniq_TCRs].tolist() # extract 'unique_TCRs' as the list n
            k = sub_df[phenotype_asso_TCRs].tolist() # extract 'phenotype_associated_TCRs' as the list k
            
            '''
            Perform optimization
            '''
            if jac == False and opt_method != 'Newton-CG': # not use jacobian (note that New-CG needs jacobian)

                # res = minimize(neg_objective,priors,args=(n,k),tol=precision,options={'maxiter':max_iterations},bounds=((0,None),(0,None)))
                res = minimize(neg_objective,priors,args=(n,k),method=opt_method,bounds=((0,None),(0,None))) 
                # note: bounds=((0,None),(0,None)) set the bounds of a and b > 0

            else: # use jacobian 
                if jac == False and opt_method == 'Newton-CG': # New-CG method needs jacobian function
                    print('Jacobian function is used in Newton-CG optimization')

                res = minimize(neg_objective,priors,jac=jacobian_fun,args=(n,k),method=opt_method,bounds=((0,None),(0,None)))
                
            return res.x
        
    
        # set total_counts, c0_counts and c1_counts
        self.total_counts = df.shape[0]
        self.c0_counts = df[df[phenotype_status]==0].shape[0]
        self.c1_counts = df[df[phenotype_status]==1].shape[0]

        df_c0 = df[df[phenotype_status]==0] # subdf of negative class
        df_c1 = df[df[phenotype_status]==1] # subdf of positive class
        df_c0.reset_index(drop=True,inplace=True) 
        df_c1.reset_index(drop=True,inplace=True)

        if self.obj_plot == True: # plotting objective function value

            # optimize priors of negative class
            obj_value = list()
            self.prior_c0 = optimize_prior(df_c0,self.prior_c0,self.opt_method,self.jac,neg_obj_jac,self.lr,self.precision,self.max_iter)

            fig = plt.figure()
            fig.suptitle('objective function plots')
            ax = fig.add_subplot(1, 2, 1)
            ax.plot(obj_value)
            ax.set_title('class_0')
            ax.set_xlabel('number of interation')
            ax.set_ylabel('objective value')

            # optimize priors of positive class
            obj_value = list()
            self.prior_c1 = optimize_prior(df_c1,self.prior_c1,self.opt_method,self.jac,neg_obj_jac,self.lr,self.precision,self.max_iter)
            ax = fig.add_subplot(1, 2, 2)
            ax.plot(obj_value)
            ax.set_title('class_1')
            ax.set_xlabel('number of interation')
            ax.set_ylabel('objective value')

        else: # not plotting 
            obj_value = list()
            self.prior_c0 = optimize_prior(df_c0,self.prior_c0,self.opt_method,self.jac,neg_obj_jac,self.lr,self.precision,self.max_iter)
            obj_value = list()
            self.prior_c1 = optimize_prior(df_c1,self.prior_c1,self.opt_method,self.jac,neg_obj_jac,self.lr,self.precision,self.max_iter)
        
 
    def priors(self):
        '''
        Return priors
        '''
        # return 'priors_c0: '+str(self.prior_c0)+' priors_c1: '+str(self.prior_c1)
        return [self.prior_c0, self.prior_c1]
        

    def predict(self,df,uniq_TCRs,phenotype_asso_TCRs):
        '''
        Predicting new data
        Arg:
            df - pandas dataframe, data to predict
            uniq_TCRs - str, the column name representing the number of unique TCRs
            phenotype_asso_TCRs - str, the column name representing the number of TCRs associated with the phenotype status
        Return:
            a list of predictions
        '''
        def predict_novel(c0_counts,c1_counts,prior_c0, prior_c1, n, k):
            '''
            Predicting a novel object
            Args:
                c0_counts - int(>=0) the number of negative samples in the training set
                c1_counts - int(>=0), the number of positive samples in the training set
                prior_c0 - [a_c0,b_c0], priors of negative class
                prior_c1 - [a_c1,b_c1], priors of positive class
                n - int(>=0), number of unique_TCRs of the novel object
                k - int(>=0), number of phenotype_associated_TCRs of the novel object
            Return:
                predicted label
            '''
            N = [c0_counts, c1_counts]

            '''
            Compute decision function
            '''
            # log-posterior odds ratio
            F = log(N[1] + 1) - log(N[0] + 1) + betaln(prior_c0[0], prior_c0[1]) - betaln(prior_c1[0],prior_c1[1]) + \
                betaln(k + prior_c1[0], n - k + prior_c1[1]) - betaln(k + prior_c0[0], n - k + prior_c0[1])

            if F <= 0:
                predict = 0
            else:
                predict = 1

            return predict

        # Check whether the model has been trained
        if self.total_counts == None:
            raise NotFittedError("This %(name)s instance is not fitted "
                                 "yet" % {'name': type(self).__name__})
        # predict each sample in the df
        # pred = [predict_novel(self.c0_counts,self.c1_counts,self.prior_c0,self.prior_c1,row[uniq_TCRs],row[phenotype_asso_TCRs]) 
        #         for _,row in df.iterrows()]

        pred = df.apply(lambda row: predict_novel(n=row[uniq_TCRs],k= row[phenotype_asso_TCRs],
                                                c0_counts=self.c0_counts,c1_counts=self.c1_counts,
                                                prior_c0=self.prior_c0,prior_c1=self.prior_c1),axis=1).values

        return pred
    
    # Posteriror probability of positive class
    def predict_proba_pos(self,df,uniq_TCRs,phenotype_asso_TCRs):
        '''
        Compute posterior probabilities of positive class (class 1)
        Arg:
            df - pandas dataframe, data to predict 
            unique_TCRs - str, the column name representing the number of unique TCRs
            phenotype_associated_TCRs - str, the column name representing the number of TCRs associated with the phenotype status        
        Return:
            a list of positive-class probabilities
        '''
        
        def pred_pos_novel(c0_counts,c1_counts, prior_c0, prior_c1, n, k):
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

            # positive-class posterior probability
            # post_prob_c1 = exp(F)/(1+exp(F)) 
            if F >= 0:
                post_prob_c1 = 1/(1+exp(-F))
            else:
                post_prob_c1 = exp(F)/(1+exp(F))

            return post_prob_c1
        
        # Check whether the model has been trained
        if self.total_counts == None:
            raise NotFittedError("This %(name)s instance is not fitted "
                                 "yet" % {'name': type(self).__name__})
        # compute each sample in the df
        # pred_prob_c1 = [pred_pos_novel(self.c0_counts,self.c1_counts,self.prior_c0,self.prior_c1,row[uniq_TCRs],row[phenotype_asso_TCRs]) 
        #                 for _,row in df.iterrows()]

        pred_prob_c1 = df.apply(lambda row: pred_pos_novel(n=row[uniq_TCRs],k= row[phenotype_asso_TCRs],
                                            c0_counts=self.c0_counts,c1_counts=self.c1_counts,
                                            prior_c0=self.prior_c0,prior_c1=self.prior_c1),axis=1).values

        return pred_prob_c1
    
    
