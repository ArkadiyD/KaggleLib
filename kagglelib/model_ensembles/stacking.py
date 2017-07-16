import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error

from ..model_predicting import predict_xgb, predict_lgbm, predict_keras, predict_sklearn
from ..model_training import train_xgb, train_lgbm, train_keras, train_sklearn

def _create_features_for_stack(train_x, y, test_x, model, task = 'multiclass_classification', n_classes = None, eval_metric = '', categorical_features = [], folds = [], train_ratio = 1.0, sparse = True, stopping_rounds = -1, epochs = 100, average_epochs = -1, seed = 1, verbose = False):
    '''
    supplementary function for stacking
    '''
    if task == 'multiclass_classification':
        preds = np.zeros((train_x.shape[0], n_classes), dtype = np.float32)
    elif task == 'binary_classification':
        preds = np.zeros((train_x.shape[0], 1), dtype = np.float32)
    elif task=='regression':
        preds = np.zeros((train_x.shape[0], 1), dtype = np.float32)
        
    new_train_x = None
    new_test_x = None

    CVs = 0
    CV_score = 0.0

    for fold in folds:
        train_index, val_index = fold[0], fold[1]

        if CVs == 0 and verbose:
            print ("Training  model : ")
            print (model)

        CVs += 1

        if sparse:
            X_train = train_x.tocsr()[train_index,:].tocoo()
            X_val = train_index.tocsr()[val_index,:].tocoo()
        elif isinstance(train_x, pd.DataFrame):
            X_train = train_x.iloc[train_index,:]
            X_val = train_x.iloc[val_index,:]
        else:
            X_train = train_x[train_index,:]
            X_val = train_x[val_index,:]
            
        Y_train = y[train_index]
        Y_val = y[val_index]

        if model.type == 'xgboost':
            model.model, score = train_xgb(X_train, Y_train, task, model.params, eval_metric, sparse = sparse, train_ratio = train_ratio, stopping_rounds = stopping_rounds, missing = np.nan, verbose = verbose, random_seed = seed)
            cur_preds = predict_xgb(X_val, model.model, missing = np.nan)
        
        elif model.type == 'lightgbm':
            model.model, score = train_lgbm(X_train, Y_train, task, model.params, categorical_features, eval_metric, train_ratio = train_ratio, stopping_rounds = stopping_rounds, verbose = True, random_seed = seed)
            cur_preds = predict_lgbm(X_val, model.model)

        elif model.type == 'keras':
            model.model, _, cur_preds = train_keras(X_train, Y_train, X_val, task, model.params, eval_metric, n_classes = None, train_ratio = train_ratio, epochs = epochs, stopping_rounds = stopping_rounds, average_epochs = average_epochs, verbose = verbose, random_seed = seed)
        
        elif model.type == 'sklearn':
            model.model = train_sklearn(X_train, Y_train, model)
            cur_preds = predict_sklearn(X_val, model.model, eval_metric, task)
        
        if len(cur_preds.shape) == 1:
            cur_preds = cur_preds.reshape(cur_preds.shape[0], 1)
        elif len(cur_preds.shape) == 2 and cur_preds.shape[1] == 2 and n_classes == 2:
            cur_preds = cur_preds[:, 1].reshape(cur_preds.shape[0], 1)
        
        preds[val_index] = cur_preds

        if eval_metric == 'auc':
            score = roc_auc_score(Y_val, cur_preds)
        elif eval_metric == 'logloss':
            score = log_loss(Y_val, cur_preds)
        elif eval_metric == 'mlogloss':
            score = log_loss(Y_val, cur_preds)
        elif eval_metric == 'error':
            score = 1 - accuracy_score(Y_val, cur_preds)
        elif eval_metric == 'merror':
            score = 1 - accuracy_score(Y_val, cur_preds) 

        elif eval_metric == 'rmse':
            score = np.sqrt(mean_squared_error(Y_val, cur_preds))
        elif eval_metric == 'mse':
            score = mean_squared_error(Y_val, cur_preds)

        CV_score += score
        if verbose:
            print ('FOLD', CVs, 'SCORE', score)
    
    
    CV_score /= float(CVs)
    print ("\tCV_score {0}\n\n".format(CV_score))
    
    if model.type == 'xgboost':
    
        model.model, score = train_xgb(train_x, y, task, model.params, eval_metric, sparse = sparse, train_ratio = train_ratio, stopping_rounds = stopping_rounds, missing = np.nan, verbose = False, random_seed = seed)
        preds_test = predict_xgb(test_x, model.model, missing = np.nan)

    elif model.type == 'lightgbm':
        
        model.model, score = train_lgbm(train_x, y, task, model.params, categorical_features, eval_metric, train_ratio = train_ratio, stopping_rounds = stopping_rounds, verbose = False, random_seed = seed)
        preds_test = predict_lgbm(test_x, model.model)

    elif model.type == 'keras':
    
        model.model, score, preds_test = train_keras(train_x, y, test_x, task, model.params, eval_metric, n_classes = None, train_ratio = train_ratio, epochs = epochs, stopping_rounds = stopping_rounds, average_epochs = average_epochs, verbose = False, random_seed = seed)
    
    elif model.type == 'sklearn':

        model.model = train_sklearn(train_x, y, model)
        preds_test = predict_sklearn(test_x, model.model, eval_metric, task)

    if len(preds_test.shape) == 1:
        preds_test = preds_test.reshape(preds_test.shape[0], 1)
    elif len(preds_test.shape) == 2 and preds_test.shape[1] == 2 and n_classes == 2:
        preds_test = preds_test[:, 1].reshape(preds_test.shape[0], 1)

    if sparse:
        new_train_x = sparse.coo_matrix(preds)
        new_test_x = sparse.coo_matrix(preds_test)
    
    else:
        new_train_x = preds.astype(np.float32)
        new_test_x = preds_test.astype(np.float32)

    return new_train_x, new_test_x, CV_score

def create_stack(train_x, train_y, test_x, models, folds, task = 'multiclass_classification', eval_metric = '', categorical_features = [], n_classes = None, train_ratio = 1.0, sparse = False, stopping_rounds = [], epochs = 100, average_epochs = -1, random_seed = 1, verbose = False):
    '''genetic algorithm for features selection
    
    creates stack features based on out-of-fold techinque
    https://mlwave.com/kaggle-ensembling-guide/

    Parameters
    ----------
    train_x : pandas.DataDrame or numpy.ndarray
        train dataset
    train_y : np.ndarray
        target
    test_x : pandas.DataDrame or numpy.ndarray
        test dataset
    models : list of mModel instances
        list of models to make stack features
    folds : list of pairs of lists
        indices of train and validation folds
        folds[i][0] - train indices in i-th train-val split
        folds[i][1] - validation indices in i-th train-val split
    task : string, 'regression' or 'binary_classification' or 'multiclass_classification'
        task to solve
    model : Model
        Model  instance
    eval_metric : string, possible variants: 'mse', rmse', 'auc', 'logloss', 'mlogloss', 'error', 'merror' 
        eval_metric for model
    categorical_features : list of strings or lists of integers, optional
        column names (if train is Pandas DataFrame) or column indices (if train is Numpy array) of categorical features
        [] by default
    n_classes : integer, optional
        number of classes in case of classification task
        2 by default
    train_ratio : float, optional
        on which part of train, validate (for early stopping) on remaining
        1.0 by default
    sparse : boolean, optional
        whether train is sparse Scipy matrix
        False by default
    stopping rounds : integer, optional
        number of early stopping rounds (or epochs for neral nets) in CV evaluations, -1 means for fixed number of rounds or epochs
        -1 by default
    epochs : integer, optional
        number of epochs in case of neural network model
        10 by default
    average_epochs : integer, optional
        number of last epochs, predictions in which are averaged, in case of neural network model
        -1 by default
    missing : integer or np.nan
        missing values for xgboost models
        np.nan by default
    seed : integer, optional
        random seed for stratification into train and validation subsets on case of early stopping in model
        1 by default
    verbose : , optional
        whether to print running info
        False by default

    Returns 
    -------
    list of best features, best CV score  (CV score with selected features)
    '''
    m = 0
    for model in models:
        if m == 0:  
        
            stack_X_train, stack_X_test, _ = _create_features_for_stack(train_x, train_y, test_x, model, task = task, n_classes = n_classes, eval_metric = eval_metric, categorical_features = categorical_features, folds = folds, train_ratio = train_ratio, sparse = sparse, stopping_rounds = stopping_rounds[m], epochs = epochs, average_epochs = average_epochs, seed = random_seed, verbose = verbose)
        
        else:
        
            cur_stack_X_train, cur_stack_X_test, _ = _create_features_for_stack(train_x, train_y, test_x, model, task = task, n_classes = n_classes, eval_metric = eval_metric, categorical_features = categorical_features, folds = folds, train_ratio = train_ratio, sparse = sparse, stopping_rounds = stopping_rounds[m], epochs = epochs, average_epochs = average_epochs, seed = random_seed, verbose = verbose)
        
            if sparse:
        
                stack_X_train = sparse.hstack([stack_X_train, sparse.coo_matrix(cur_stack_X_train)])
                stack_X_test = sparse.hstack([stack_X_test, sparse.coo_matrix(cur_stack_X_test)])
        
            else:
                stack_X_train = np.hstack([stack_X_train, cur_stack_X_train])
                stack_X_test = np.hstack([stack_X_test, cur_stack_X_test])

        if verbose:
            print('X for stacking shape', stack_X_train.shape, stack_X_test.shape)
            print stack_X_train, stack_X_test

        m += 1

    return stack_X_train, stack_X_test
