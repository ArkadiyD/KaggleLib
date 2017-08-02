import xgboost as xgb
import lightgbm as lgbm 
import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error
from keras.callbacks import EarlyStopping

from ..model_predicting import *
from ..model_training import *

from ..utils import *

def CV_score_xgb(train_x, train_y, params, eval_metric, folds, sparse = False, stopping_rounds = -1, missing = np.nan, verbose = False):
    """calculates score on cross-validation for xgboost models
    
    Parameters
    ----------
    train : numpy.ndarray or scipy.sparse array
        train dataset
    train_y : numpy.ndarray
        target
    params: dictionary
        xgbosot model parameters
    folds : list of pairs of lists
        indices of train and validation folds for encoding
        folds[i][0] - train indices in i-th train-val split
        folds[i][1] - validation indices in i-th train-val split
    sparse : boolean, optional
        whether train is sparse Scipy matrix
        False by default
    stopping rounds : integer, optional
        number of early stopping rounds in xgboost CV evaluations, -1 means no early stopping
        -1 by default
    missing : integer or np.nan, optional
        missing values for xgboost models
        np.nan by default
    verbose : boolean, optional:
        whether to print running info
        False by default
    
    Returns 
    -------
    cross-validation score
    """

    CV_score = 0.0

    for f in xrange(len(folds)):

        fold = folds[f]
        train_index, val_index = fold[0], fold[1]

        if f == 0 and verbose:
            
            print ("Training with params : ")
            print (params)
        
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

        if stopping_rounds < 0:
            model = xgb.train(params, dtrain, int(params['rounds']))
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
            model = xgb.train(params, dtrain, 1000, evals = watchlist, verbose_eval = verbose, early_stopping_rounds = stopping_rounds)
            score = model.best_score

        
        CV_score += score

        if verbose:
            print ('FOLD', f, 'SCORE', score)

    CV_score /= float(len(folds))

    if verbose:
        print '\tCV_score {0}\n\n'.format(CV_score)

    return CV_score

def CV_score_lgbm(train_x, train_y, params, categorical, eval_metric, folds, stopping_rounds = -1, verbose = False):
    """calculates score on cross-validation for lightgbm models
    
    Parameters
    ----------
    train : Pandas DataFrame or numpy array
        train dataset
    train_y : Numpy array
        target
    params : dictionary
        lightgbm model parameters
    categorical : list of strings or lists of integers
        column names (if train is Pandas DataFrame) or column indices (if train is Numpy array) of categorical features
    eval_metric : string, possible variants: 'mse', rmse', 'auc', 'logloss', 'mlogloss', 'error', 'merror' 
        eval_metric for lightgbm
    folds : list of pairs of lists
        indices of train and validation folds for encoding
        folds[i][0] - train indices in i-th train-val split
        folds[i][1] - validation indices in i-th train-val split
    stopping rounds : integer, optional
        number of early stopping rounds in xgboost CV evaluations, -1 means no early stopping
        -1 by default
    verbose : boolean, optional:
        whether to print running info
        False by default
    
    Returns 
    -------
    cross-validation score
    """    

    CV_score = 0.0

    for f in xrange(len(folds)):

        fold = folds[f]
        train_index, val_index = fold[0], fold[1]

        if f == 0 and verbose:
            
            print ("Training with params : ")
            print (params)
        
        y_train = train_y[train_index]
        y_val = train_y[val_index]
    
        if isinstance(train_x, pd.DataFrame):
            X_train = train_x.ix[train_index]
            X_val = train_x.ix[val_index]       
        else:
            X_train = train_x[train_index]
            X_val = train_x[val_index]

        train_data = lgbm.Dataset(X_train, label = y_train, feature_name = 'auto', categorical_feature = categorical)
        valid_data = lgbm.Dataset(X_val, label = y_val, feature_name = 'auto', categorical_feature = categorical)
            
                     
        params['max_depth'] = int(params['max_depth'])
        params['num_leaves'] = int(params['num_leaves'])
        params['bagging_freq'] = int(params['bagging_freq'])
        params['max_bin'] = int(params['max_bin'])
        params['min_data_in_leaf'] = int(params['min_data_in_leaf'])

        if stopping_rounds < 0:
            params['num_boost_round'] = int(params['num_boost_round'])
            model = lgbm.train(params, train_data)
            preds_val = model.predict(X_val)

        else:
            model = lgbm.train(params, train_data, num_boost_round = 1000, valid_sets = valid_data, verbose_eval = verbose, early_stopping_rounds = stopping_rounds)
            preds_val = model.predict(X_val, num_iteration = model.best_iteration)
            
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

        CV_score += score

        if verbose:
            print ('FOLD', f, 'SCORE', score)

    CV_score /= float(len(folds))

    if verbose:
        print '\tCV_score {0}\n\n'.format(CV_score)

    return CV_score

def CV_score_sklearn(train_x, train_y, model, task, eval_metric, folds, verbose = False):
    """calculates score on cross-validation for sklearn models
    
    Parameters
    ----------
    train : Pandas DataFrame or numpy.array or Scipy.sparse array
        train dataset
    train_y : Numpy array
        target
    model : Model
        instance of Model with type 'sklearn'
    task : string, 'regression' or 'binary_classification' or 'multiclass_classification'
        task to solve
    eval_metric : string, possible variants: 'mse', rmse', 'auc', 'logloss', 'mlogloss', 'error', 'merror' 
        eval_metric
    folds : list of pairs of lists
        indices of train and validation folds for encoding
        folds[i][0] - train indices in i-th train-val split
        folds[i][1] - validation indices in i-th train-val split
    verbose : boolean, optional:
        whether to print running info
        False by default
    
    Returns 
    -------
    cross-validation score
    """    

    CV_score = 0.0

    for f in xrange(len(folds)):

        fold = folds[f]
        train_index, val_index = fold[0], fold[1]

        if f == 0 and verbose:
            
            print ("Training with params : ")
            print (model.params)
        
        if isinstance(train_x, pd.DataFrame):
            X_train = train_x.ix[train_index]
            X_val = train_x.ix[val_index]
        else:
            X_train = train_x[train_index]
            X_val = train_x[val_index]
            
        y_train = train_y[train_index]
        y_val = train_y[val_index]
        
        model.model.set_params(**model.params)
        model.model = model.model.fit(X_train, y_train)

        preds_val = predict_sklearn(X_val, model.model, eval_metric, task)

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

        CV_score += score

        if verbose:
            print ('FOLD', f, 'SCORE', score)

    CV_score /= float(len(folds))

    if verbose:
        print '\tCV_score {0}\n\n'.format(CV_score)

    return CV_score

def CV_score_keras(train_x, train_y, task, params, eval_metric, folds, n_classes = None, epochs = 1000, stopping_rounds = -1, average_epochs = -1, verbose = False):
    """calculates score on cross-validation for keras models
    
    Parameters
    ----------
    train_x : Pandas DataFrame or numpy.array or Scipy.sparse array
        train dataset
    train_y : Numpy array
        target
    task : string, 'regression' or 'binary_classification' or 'multiclass_classification'
        task to solve
    params : dictionary
        keras model parameters
    eval_metric : string, possible variants: 'mse', rmse', 'auc', 'logloss', 'mlogloss', 'error', 'merror' 
        eval_metric
    folds : list of pairs of lists
        indices of train and validation folds for encoding
        folds[i][0] - train indices in i-th train-val split
        folds[i][1] - validation indices in i-th train-val split
    n_classes : integer, optional
        number of classes in cases of classification task
        2 by default
    epochs : integer, optional
        number of epoches
        1000 by default
    stopping rounds : integer, optional
        number of early stopping rounds in xgboost CV evaluations, -1 means no early stopping
        -1 by default
    average_epochs : integer
        predictions on last average_epochs epochs are averaged for validation and test datasets, -1 means no averaging
        -1 by default
    verbose : boolean, optional:
        whether to print running info
        False by default
    
    Returns 
    -------
    cross-validation score
    """    

    CV_score = 0.0

    for f in xrange(len(folds)):

        fold = folds[f]
        train_index, val_index = fold[0], fold[1]

        if f == 0 and verbose:
            
            print ("Training with params : ")
            print (params)
        
        if isinstance(train_x, pd.DataFrame):
            X_train = train_x.ix[train_index]
            X_val = train_x.ix[val_index]
        else:
            X_train = train_x[train_index]
            X_val = train_x[val_index]
            
        y_train = train_y[train_index]
        y_val = train_y[val_index]
        
        if eval_metric == 'error' or eval_metric == 'merror':
            monitor = 'val_acc'
        else:
            monitor = 'val_loss'

        history_callback = HistoryCallback(X_val)
        
        model = generate_keras_model(params, task = task, n_classes = n_classes, verbose = verbose)

        if stopping_rounds >= 0:
            early_stopping = EarlyStopping(monitor = monitor, patience = stopping_rounds, verbose = verbose, mode = 'auto')
            model.fit(X_train, y_train, batch_size = 128, epochs = 1000, verbose = verbose, callbacks = [early_stopping, history_callback], validation_data = (X_val, y_val))
        else:
            model.fit(X_train, y_train, batch_size = 128, epochs = epochs, verbose = verbose, validation_data = (X_val, y_val), callbacks = [history_callback])
        
        if eval_metric == 'auc':
            epochs_metrics = [roc_auc_score(y_val, history_callback.val_predictions[i]) for i in xrange(len(history_callback.val_predictions))]
        elif eval_metric == 'logloss' or eval_metric == 'mlogloss':
            epochs_metrics = [log_loss(y_val, history_callback.val_predictions[i]) for i in xrange(len(history_callback.val_predictions))]
        elif eval_metric == 'error':
            epochs_metrics = [accuracy_score(y_val, history_callback.val_predictions[i]) for i in xrange(len(history_callback.val_predictions))]
        elif eval_metric == 'mse':
            epochs_metrics = [mean_squared_error(y_val, history_callback.val_predictions[i]) for i in xrange(len(history_callback.val_predictions))]
        elif eval_metric == 'rmse':
            epochs_metrics = [np.sqrt(mean_squared_error(y_val, history_callback.val_predictions[i])) for i in xrange(len(history_callback.val_predictions))]
        else:
            raise ValueError("eval_metric should be 'auc' or 'logloss' or 'mlogloss' or 'error' or 'merror' or 'mse' or 'rmse', got %s" % eval_metric)

        if average_epochs < 0: 

            if eval_metric == 'auc':
                best_epoch = np.argmax(epochs_metrics)
            else:
                best_epoch = np.argmin(epochs_metrics)
                
            preds_val = history_callback.val_predictions[best_epoch]
            
        else:
            if eval_metric == 'auc':
                best_epoches = np.argsort(epochs_metrics)[-average_epochs:]
            else:
                best_epoches = np.argsort(epochs_metrics)[:average_epochs]
            
            history_callback.val_predictions = np.array(history_callback.val_predictions)
            preds_val = np.mean(history_callback.val_predictions[best_epoches], axis = 0)
        
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

        CV_score += score

        if verbose:
            print ('FOLD', f, 'SCORE', score)

    CV_score /= float(len(folds))

    if verbose:
        print '\tCV_score {0}\n\n'.format(CV_score)

    return CV_score