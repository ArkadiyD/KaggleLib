import numpy as np
import pandas as pd

def logarithm(train, test, colnames):
    '''create column transformation as logarithm(x) or logarithm(x + 1)

    Parameters
    ----------
    train : Pandas Dataframe
        train dataset
    test : Pandas Dataframe
        test dataset
    colnames : list of string
        column names to make logarithm transformation

    Returns
    -------
    list of names of created columns
    '''

    new_cols = []
    for colname in colnames:
        if np.min(train[colname]) > 0:  
            train.loc[:, 'log_' + colname] = np.log(train[colname])
            test.loc[:, 'log_' + colname] = np.log(test[colname])

        elif np.min(train[colname]) > -1:   
            train.loc[:, 'log_' + colname] = np.log(1 + train[colname])
            test.loc[:, 'log_' + colname] = np.log(1 + test[colname])

        new_cols.append('log_' + colname)

    return new_cols

def exponent(train, test, colnames, type):
    '''create exponent transformation as exp(x) or exp(-x)

    Parameters
    ----------
    train : Pandas Dataframe
        train dataset
    test : Pandas Dataframe
        test dataset
    colnames : list of string
        column names to make exponent transformation
    type : str, 'plus' or 'minus'
        'plus' means exp(x), 'minus' means exp(-x)

    Returns
    -------
    list of names of created columns
    '''

    new_cols = []
    for colname in colnames:
        
        train.loc[:, 'exp_' + colname] = np.exp(train[colname])
        test.loc[:, 'exp_' + colname] = np.exp(test[colname])
        new_cols.append('exp_' + colname)

        if minus_exp:
            train.loc[:, 'minus_exp_' + colname] = np.exp(-train[colname])
            test.loc[:, 'minus_exp_' + colname] = np.exp(-test[colname])
            new_cols.append('minus_exp_' + colname)

    return new_cols
    
def sigmoid(train, test, colnames):
    '''create sigmoid transformation as 1 / (1 + exp(-x))

    Parameters
    ----------
    train : Pandas Dataframe
        train dataset
    test : Pandas Dataframe
        test dataset
    colnames : list of string
        column names to make exponent transformation

    Returns
    -------
    list of names of created columns
    '''
    new_cols = []
    for colname in colnames:
        
        train.loc[:, 'sigmoid_' + colname] = 1.0 / (1.0 + np.exp(-train[colname]))
        test.loc[:, 'sigmoid_' + colname] = 1.0 / (1.0 + np.exp(-test[colname]))
    
        new_cols.append('exp_' + colname)

    return new_cols
        
def trigonometry(train, test, colnames):
    '''creates trigonometric transformation as sin(x), cos(x), tan(x)

    Parameters
    ----------
    train : Pandas Dataframe
        train dataset
    test : Pandas Dataframe
        test dataset
    colnames : list of string
        column names to make exponent transformation

    Returns
    -------
    list of names of created columns
    '''

    new_cols = []
    for colname in colnames:
        
        train.loc[:, 'sin_' + colname] = np.sin(train[colname])
        test.loc[:, 'sin_' + colname] = np.sin(test[colname])

        train.loc[:, 'cos_' + colname] = np.cos(train[colname])
        test.loc[:, 'cos_' + colname] = np.cos(test[colname])

        train.loc[:, 'tan_' + colname] = np.tan(train[colname])
        test.loc[:, 'tan_' + colname] = np.tan(test[colname])
        new_cols.append('sin_' + colname)
        new_cols.append('cos_' + colname)
        new_cols.append('tan_' + colname)

    return new_cols
    