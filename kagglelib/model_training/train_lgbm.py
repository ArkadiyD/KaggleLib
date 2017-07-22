import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

def train_lgbm(train_x, train_y, task, params, categorical, eval_metric, train_ratio = 0.9, stopping_rounds = -1, verbose = False, random_seed = 1):
    """calculates score on cross-validation for lightgbm models
    
    Parameters
    ----------
    train : pandas.DataFrame or numpy.ndarray
        train dataset
    train_y : numpy.ndarray
        target
    task : string, 'regression' or 'binary_classification' or 'multiclass_classification'
        task to solve
    params : dictionary
        lightgbm model parameters
    categorical : list of strings or lists of integers
        column names (if train is Pandas DataFrame) or column indices (if train is Numpy array) of categorical features
    eval_metric : string, possible variants: 'mse', rmse', 'auc', 'logloss', 'mlogloss', 'error', 'merror' 
        eval_metric for lightgbm
    train_ratio : float
        train part of size train_ratio * train.shape[0] is used for training, remaining part is used for validation and early stopping
        0.9 by default
    stopping rounds : integer
        number of early stopping rounds in xgboost CV evaluations, -1 means no early stopping
        1 by default
    verbose : boolean, optional:
        whether to print running info
        False by default
    random_seed: integer, optional
        random seed for on shuffle split for train-validation
        1 by default

    Returns 
    -------
    trained lightgbm model, validation score
    """    

    if verbose:
        print ("Training with params : ")
        print (params)

    if train_ratio < 1:
        if task == 'binary_classification' or task == 'multiclass_classification':
            shuffler = StratifiedShuffleSplit(n_splits = 1, test_size = 1 - train_ratio, random_state = random_seed)
        elif task == 'regression':
            shuffler = ShuffleSplit(n_splits = 1, test_size = 1 - train_ratio, random_state = random_seed)
        else:
            raise ValueError("task should be 'regression' or 'binary_classification' or 'multiclass_classification', got %s" % task)

        for train_index, val_index in shuffler.split(train_x, train_y):
            break
    
        print 'train_size: ', train_index.shape[0], ' -- val_size: ', val_index.shape[0]

    else:
        train_index = np.arange(train_x.shape[0])
        val_index = []

    y_train = train_y[train_index]
    y_val = train_y[val_index]

    if isinstance(train_x, pd.DataFrame):
        X_train = train_x.iloc[train_index,:]
        X_val = train_x.iloc[val_index,:]
    
        train_data = lgb.Dataset(X_train.values, label = y_train, free_raw_data = True, feature_name = 'auto', categorical_feature = categorical)
        valid_data = lgb.Dataset(X_val.values, label = y_val, free_raw_data = True, feature_name = 'auto', categorical_feature = categorical)
     
    else:
        X_train = train_x[train_index]
        X_val = train_x[val_index]
        
        train_data = lgb.Dataset(X_train, label = y_train, free_raw_data = True, feature_name = 'auto', categorical_feature = categorical)
        valid_data = lgb.Dataset(X_val, label = y_val, free_raw_data = True, feature_name = 'auto', categorical_feature = categorical)
     
    params['max_depth'] = int(params['max_depth'])
    params['num_leaves'] = int(params['num_leaves'])
    params['bagging_freq'] = int(params['bagging_freq'])
    params['max_bin'] = int(params['max_bin'])
    params['min_data_in_leaf'] = int(params['min_data_in_leaf'])

    if 'num_boost_round' in params:
        params['num_boost_round'] = int(params['num_boost_round'])
        model = lgb.train(params, train_data)
    
    else:
        model = lgb.train(params, train_data, num_boost_round = 1000, valid_sets = [valid_data], verbose_eval = verbose, early_stopping_rounds = stopping_rounds)

    if train_ratio < 1:
        if isinstance(X_val, pd.DataFrame):
            preds_val = model.predict(X_val.values, num_iteration = -1)
        else:
            preds_val = model.predict(X_val, num_iteration = -1)
        
        if verbose:
            print model.best_iteration

        if eval_metric == 'auc':
            score = roc_auc_score(y_val, preds_val)
        elif eval_metric == 'logloss':
            score = log_loss(y_val, preds_val)
        elif eval_metric == 'mlogloss':
            score = log_loss(y_val, preds_val)
        elif eval_metric == 'error':
            score = 1 - accuracy_score(y_val, preds_val)
        elif eval_metric == 'merror':
            score = 1 - accuracy_score(y_val, preds_val) 

        elif eval_metric == 'rmse':
            score = np.sqrt(mean_squared_error(y_val, preds_val))
        elif eval_metric == 'mse':
            score = mean_squared_error(y_val, preds_val)
        else:
            raise ValueError("eval_metric should be 'auc' or 'logloss' or 'mlogloss' or 'error' or 'merror' or 'mse' or 'rmse', got %s" % eval_metric)

    else:
        score = None

    return model, score