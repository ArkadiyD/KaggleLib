import xgboost as xgb
import numpy as np
from scipy import sparse
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error

def train_xgb(train_x, train_y, task, params, eval_metric, sparse = False, train_ratio = 0.9, stopping_rounds = -1, missing = np.nan, verbose = False, random_seed = 1):
    """trains xgboost models
    
    Parameters
    ----------
    train_x : numpy.ndarray or scipy.sparse matrix
        train dataset
    train_y : numpy.ndarray
        target
    task : string, 'regression' or 'binary_classification' or 'multiclass_classification'
        task to solve
    params: dictionary
        xgbosot model parameters
    sparse : boolean
        whether train is scipy.sparse matrix
        False by default
    train_ratio : float
        train part of size train_ratio * train.shape[0] is used for training, remaining part is used for validation and early stopping
        0.9 by default
    stopping rounds : integer
        number of early stopping rounds in xgboost CV evaluations, -1 means no early stopping
        -1 by default
    missing : float or np.nan
        missing values for xgboost models
        np.nan by default
    verbose : boolean, optional:
        whether to print running info
        False by default
    random_seed: integer, optional
        random seed for on shuffle split for train-validation
        1 by default

    Returns 
    -------
    trained xgboost model, validation score
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

        for train_index, val_index in shuffler.split(np.zeros(train_y.shape[0]), train_y):
            break
    
        print 'train_size: ', train_index.shape[0], ' -- val_size: ', val_index.shape[0]

    else:
        train_index = np.arange(train_x.shape[0])
        val_index = []

    
    
    if sparse:

        X_train = train_x.tocsr()[train_index,:].tocoo()
        X_val = train_x.tocsr()[val_index,:].tocoo()
        y_train = train_y[train_index]
        y_val = train_y[val_index]

        dtrain = xgb.DMatrix(X_train.tocsc(), y_train, missing = missing)
        dvalid = xgb.DMatrix(X_val.tocsc(), y_val, missing = missing)

    else:

        X_train = train_x[train_index,:]
        X_val = train_x[val_index,:]
        y_train = train_y[train_index]
        y_val = train_y[val_index]
    
        dtrain = xgb.DMatrix(X_train, y_train, missing = missing)
        dvalid = xgb.DMatrix(X_val, y_val, missing = missing)
    

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    
    if 'max_depth' in params:
        params['max_depth'] = int(params['max_depth'])
    
    if 'rounds' in params:
        model = xgb.train(params, dtrain, int(params['rounds']))
        
        if train_ratio >= 1.0:
            score = None
        else:
            preds_val = model.predict(dvalid)
            
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
        model = xgb.train(params, dtrain, 1000, evals = watchlist, verbose_eval = verbose, early_stopping_rounds = stopping_rounds)
        score = model.best_score
    

    return model, score