import lightgbm as lgb
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error
import pandas as pd

def find_params_lgbm(train_x, train_y, task, categorical, eval_metric, n_classes, folds, total_evals = 50, stopping_rounds = -1, verbose = False):
    '''find optimal parameters on cross-validation for lightgbm by Hyperopt
    
    Parameters
    ----------
    train : pandas.DataFrame or numpy.ndarray
        train dataset
    train_y : numpy.ndarray
        target
    task : string, 'regression' or 'binary_classification' or 'multiclass_classification'
        task to solve
    categorical : list of strings or lists of integers
        column names (if train is Pandas DataFrame) or column indices (if train is Numpy array) of categorical features
    eval_metric : string, possible variants: 'mse', rmse', 'auc', 'logloss', 'mlogloss', 'error', 'merror' 
        eval_metric for lightgbm
    n_classes : 2
        number of classes in case of classification task
    folds : list of pairs of lists
        indices of train and validation folds for encoding
        folds[i][0] - train indices in i-th train-val split
        folds[i][1] - validation indices in i-th train-val split
    total_evals : integer
        number of total evaluations of lightgbm models, number of parameters sets to probe is total_evals / len(folds)
        50 by default
    stopping rounds : integer
        number of early stopping rounds in lightgbm CV evaluations
        1 by default
    verbose : boolean, optional:
        whether to print running info
        False by default
    
    Returns 
    -------
    dictionary of best parameters on cross-validation
    '''

    np.random.seed(777)
    
    if eval_metric == 'logloss':
        metric = 'binary_logloss'
    elif eval_metric == 'mlogloss':
        metric = 'multi_logloss'
    elif eval_metric == 'error':
        metric = 'binary_error'
    elif eval_metric == 'merror':
        metric = 'multi_error'

    def score(params):
        
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

            if stopping_rounds < 0:
                params['num_boost_round'] = int(params['num_boost_round'])
                if verbose:
                    print train_data.feature_name, categorical
                model = lgb.train(params, train_data)
                if isinstance(X_val, pd.DataFrame):
                    preds_val = model.predict(X_val.values)
                else:
                    preds_val = model.predict(X_val)

            else:
                model = lgb.train(params, train_data, num_boost_round = 1000, valid_sets = valid_data, verbose_eval = verbose, early_stopping_rounds = stopping_rounds)
                if isinstance(X_val, pd.DataFrame):
                    preds_val = model.predict(X_val.values, num_iteration = model.best_iteration)
                else:
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
            print params

        if params['metric'] != 'auc':
            return {'loss': CV_score, 'status': STATUS_OK}
        else:
            return {'loss': -CV_score, 'status': STATUS_OK}
            
    def optimize(trials):
        

        if task == 'regression':
            objective = 'regression'
        elif task == 'binary_classification':
            objective = 'binary'
        elif task == 'multiclass_classification':
            objective = 'multiclass'

        space = {
            'task': 'train',
            'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
            'objective': objective,
            'metric': eval_metric,
            'num_leaves': hp.quniform('num_leaves', 30, 150, 5),
            'max_depth': hp.quniform('max_depth', 3, 15, 1),
            'feature_fraction': hp.uniform('feature_fraction', 0.6, 0.97),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 0.95),
            'bagging_freq': hp.quniform('bagging_freq', 3, 20, 1),
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 100, 1),
            'learning_rate': hp.uniform('learning_rate', 0.003, 0.3),
            'max_bin' : hp.quniform('max_bin', 20, 1000, 10),
            'lambda_l1' : hp.loguniform('lambda_l1', -3, 1),
            'lambda_l2' : hp.loguniform('lambda_l2', -3, 1),
            'min_gain_to_split' : hp.uniform('min_gain_to_split', 0, 1),
            'verbose': -1
        }

        if stopping_rounds < 0:
            space['num_boost_round'] = hp.quniform('num_boost_round', 30, 500, 5)

        if task == 'multiclass_classification':
            space['num_class'] = n_classes

        best = fmin(score, space, algo = tpe.suggest, trials = trials, max_evals = int(total_evals / len(folds)))
        for key in space:
            if key in best:
                space[key] = best[key]
        space['boosting_type'] = ['gbdt', 'dart'][best['boosting_type']]

        if verbose:
            print ('-'*50,'BEST PARAMS',space)
        return space

    trials = Trials()
    best = optimize(trials)
    
    losses = abs(np.array(trials.losses()))
    if eval_metric == 'auc':
        best_score = np.max(losses)
    else:
        best_score = np.min(losses)

    if verbose:
    
        print losses
        print 'BEST CV SCORE: ', best_score

    return best, best_score

