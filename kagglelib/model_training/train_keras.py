import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from keras.layers import Dense, Dropout,Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import keras.backend as K

from ..utils import *

def train_keras(train_x, train_y, test_x, task, params, eval_metric, n_classes = 2, train_ratio = 0.9, epochs = 1000, stopping_rounds = -1, average_epochs = -1, random_seed = 1, verbose = False):
    '''calculates score on cross-validation for keras models
    
    Parameters
    ----------
    train : pandas.DataFrame or numpy.ndarray
        train dataset
    train_y : numpy.ndarray
        target
    task : string, 'regression' or 'binary_classification' or 'multiclass_classification'
        task to solve
    params : dictionary
        keras model parameters
    categorical : list of strings or lists of integers
        column names (if train is Pandas DataFrame) or column indices (if train is Numpy array) of categorical features
    eval_metric : string, possible variants: 'mse', rmse', 'auc', 'logloss', 'mlogloss', 'error', 'merror' 
        eval_metric for lightgbm
    n_classes : integer, optional
        number of classes in cases of classification task
        2 by default
    train_ratio : float, optional
        train part of size train_ratio * train.shape[0] is used for training, remaining part is used for validation and early stopping
        0.9 by default
    epochs : integer, optional
        number of epoches
        1000 by default
    stopping rounds : integer, optional
        number of early stopping rounds in xgboost CV evaluations, -1 means no early stopping
        -1 by default
    average_epochs : integer
        predictions on last average_epochs epochs are averaged for validation and test datasets, -1 means no averaging
        -1 by default
    random_seed: integer, optional
        random seed for on shuffle split for train-validation
        1 by default
    verbose : boolean, optional:
        whether to print running info
        False by default

    Returns 
    -------
    trained keras model, validation score, (optional) predictions on test
    '''    

    K.clear_session()
    
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


    X_train = train_x[train_index]
    X_val = train_x[val_index]
        
    y_train = train_y[train_index]
    y_val = train_y[val_index]
    
    if eval_metric == 'error' or eval_metric == 'merror':
        monitor = 'val_acc'
    else:
        monitor = 'val_loss'

    model = generate_keras_model(params, task = task, n_classes = n_classes, verbose = verbose)
    history_callback = HistoryCallback(X_val, test_x)

    if stopping_rounds >= 0:
        early_stopping = EarlyStopping(monitor = monitor, patience = stopping_rounds, verbose = verbose, mode = 'auto')
        model.fit(X_train, y_train, batch_size = 128, epochs = 1000, verbose = verbose, callbacks = [early_stopping, history_callback], validation_data = (X_val, y_val))
            
    else:
        model.fit(X_train, y_train, batch_size = 128, epochs = epochs, callbacks = [history_callback], verbose = verbose)
    

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
        if isinstance(test_x, np.ndarray):  
            preds_test = history_callback.test_predictions[best_epoch]

    else:
        if eval_metric == 'auc':
            best_epoches = np.argsort(epochs_metrics)[-average_epochs:]
        else:
            best_epoches = np.argsort(epochs_metrics)[:average_epochs]
        
        history_callback.val_predictions = np.array(history_callback.val_predictions)
        preds_val = np.mean(history_callback.val_predictions[best_epoches], axis = 0)
    
        if isinstance(test_x, np.ndarray):
            history_callback.test_predictions = np.array(history_callback.test_predictions)
            preds_test = np.mean(history_callback.test_predictions[best_epoches], axis = 0)

    score = None
    if len(val_index) > 0:
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


    if not isinstance(test_x, np.ndarray):
        return model, score
    return model, score, preds_test