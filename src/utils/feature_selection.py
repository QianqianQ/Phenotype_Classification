# Import libraries

# Helpers
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ttest_ind

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
