import numpy as np
import pandas as pd
from collections import Counter
from ..utils import make_folds

def categorical_target_encoding(train_x, train_y, test_x, colnames_to_encode, folds, simple_scheme = True, inner_n_folds = 3, inner_seed = 1, inner_type = '', threshold = 1, alpha = 1, verbose = False):
    '''encoding categorical features with wighted target value
    
    samples with value X of a categorical feature are encoded with value (cur_mean * K + alpha * overall_mean) / (K + alpha):
        cur_mean - mean target for samples with value X for current feature
        overall_mean - mean target of overall dataset
        K - number of occurences of value X of current feature
        alpha - parameter, the larger it is, the more occurences of feature value is needed to be encoded significantly different from overall mean target  
    simple scheme means make target encoding for validation parts bases on train parts (simple out-of-fold technique)
    not simple scheme means make target encoding for train part based on k-fold split inside it (using inner_folds)
    
    Parameters
    ----------
    train_x : pandas.DataDrame
        train dataset
    train_y : np.ndarray
        target
    test_x : Pandas Dataframe
        test dataset
    colnames_to_encode : list of string
        column names to encode with weighted target
    folds: list of pairs of lists
        indices of train and validation folds for encoding
        folds[i][0] - train indices in i-th train-val split
        folds[i][1] - validation indices in i-th train-val split
        validation indices in i-th inner train-val split
        [] by default
    inner_n_folds : integer, optional
        number of folds in inner k-folds split of train part (user only if simple_scheme = False)
        3 by default
    inner_seed : integer, optional
        random seed for inner k-folds split of train part (user only if simple_scheme = False)
        1 by default
    inner_type : string, optional, 'stratified' or 'random'
        type inner k-folds split of train part (user only if simple_scheme = False)
        '' by default
    threshold : integer, optional.
        values of target which occur times less than threshold are encoded with mean target over all dataset 
        1 by default
    alpha: integer, optional
        parameter, the larger it is, the more occurences of feature value is needed to be encoded significantly different from overall mean target  
        1 by default
    verbose:boolean, optional:
        whether to print running info
        False by default

    Returns 
    -------
    list of names of created columns
    '''
    
    new_column_train_values = np.zeros((train_x.shape[0]))
    new_column_test_values = np.zeros((test_x.shape[0]))

    global_mean_y = np.mean(train_y)
    new_colnames = []

    for col_name in colnames_to_encode:
    
        new_col_name = 'mean_' + col_name
        new_colnames.append(new_col_name)
        
        if verbose:
            print new_col_name, colnames_to_encode
    
        train_x.loc[:, new_col_name] = global_mean_y
        test_x.loc[:, new_col_name] = global_mean_y

        for fold in folds:

            train_index, val_index = fold[0], fold[1]
            
            if simple_scheme:
                counter = Counter(train_x[col_name].values[train_index])
                if verbose:
                    print counter

                for value in counter:

                    if counter[value] < threshold:
                        continue

                    ind = np.where(train_x[col_name].values[train_index] == value)[0]
                    cur_y = train_y[train_index[ind]]
                    cur_mean_y = (np.mean(cur_y) * cur_y.shape[0] + alpha * global_mean_y) / (cur_y.shape[0] + alpha)

                    ind = np.where(train_x[col_name].values[val_index] == value)[0]
                    new_column_train_values[val_index[ind]] = cur_mean_y
                    
                    if verbose:
                        print value, cur_mean_y
            else:
                inner_folds = make_folds(train_y[train_index], n_folds = inner_n_folds, random_seed = inner_seed, type = inner_type)

                for inner_fold in inner_folds:

                    inner_train_index, inner_val_index = inner_fold[0], inner_fold[1]
                    
                    counter = Counter(train_x[col_name].values[train_index[inner_train_index]])
                    if verbose:
                        print counter

                    for value in counter:

                        if counter[value] < threshold:
                            continue

                        ind = np.where(train_x[col_name].values[train_index[inner_train_index]] == value)[0]
                        cur_y = train_y[train_index[inner_train_index[ind]]]
                        cur_mean_y = (np.mean(cur_y) * cur_y.shape[0] + alpha * global_mean_y) / (cur_y.shape[0] + alpha)

                        ind = np.where(train_x[col_name].values[train_index[inner_val_index]] == value)[0]
                        new_column_train_values[train_index[inner_val_index[ind]]] = cur_mean_y
                        
                        if verbose:
                            print value, cur_mean_y

        counter = Counter(train_x[col_name].values)
        if verbose:
            print counter

        for value in counter:
    
            if counter[value] < threshold:
                continue
                
            ind = np.where(train_x[col_name].values == value)[0]
            cur_y = train_y[ind]
            cur_mean_y = (np.mean(cur_y) * cur_y.shape[0] + alpha * np.mean(train_y)) / (cur_y.shape[0] + alpha)
            
            ind = np.where(test_x[col_name].values == value)[0]
            new_column_test_values[ind] = cur_mean_y

            if verbose:
                print value, cur_mean_y

        train_x.loc[:, new_col_name] = np.copy(new_column_train_values)
        test_x.loc[:, new_col_name] = np.copy(new_column_test_values)

    return new_colnames