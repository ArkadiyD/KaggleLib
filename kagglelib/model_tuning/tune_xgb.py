import xgboost as xgb
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error


def find_params_xgb(train_x, train_y, task, type, eval_metric, n_classes, folds, total_evals = 50, sparse = False, stopping_rounds = -1, missing = np.nan, verbose = False):
    '''find optimal parameters on cross-validation for xgboost by Hyperopt
    
    Parameters
    ----------
    train : numpy.ndarray or scipy.sparse array
        train dataset
    train_y : numpy.ndarray
        target
    task : string, 'regression' or 'binary_classification' or 'multiclass_classification'
        task to solve
    type : string, 'linear' or 'tree'
        type of xgboost model
    eval_metric : string, possible variants: 'rmse', 'auc', 'logloss', 'mlogloss', 'error', 'merror' 
        eval_metric for xgboost
    n_classes: 2
        numbe r of classes in case of classification task
    folds : list of pairs of lists
        indices of train and validation folds for encoding
        folds[i][0] - train indices in i-th train-val split
        folds[i][1] - validation indices in i-th train-val split
    total_evals : integer
        number of total evaluations of xgboost models, number of parameters sets to probe is total_evals / len(folds)
        50 by default
    sparse : boolean
        whether train is sparse Scipy matrix
        False by default
    stopping rounds : integer
        number of early stopping rounds in xgboost CV evaluations, -1 means for fixed number of rounds
        1 by default
    missing : integer or np.nan
        missing values for xgboost models
        np.nan by default
    verbose : , optional:
        whether to print running info
        False by default
    
    Returns 
    -------
    dictionary of best parameters on cross-validation
    '''

    np.random.seed(777)

    def score(params):
        
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
            print params

        if params['eval_metric'] != 'auc':
            return {'loss': CV_score, 'status': STATUS_OK}
        else:
            return {'loss': -CV_score, 'status': STATUS_OK}
            
    def optimize(trials):
        
        if task == 'regression':
            objective = 'reg:linear'
        elif task == 'binary_classification':
            objective = 'binary:logistic'
        elif task == 'multiclass_classification':
            objective = 'multi:softprob'

        if type == 'linear':
            
            space = {
                    'eta': hp.uniform('eta', 0.01, 0.51),
                    'lambda' : hp.uniform('lambda', 0.05, 0.7),
                    'lambda_bias' : hp.uniform('lambda_bias', 0.05, 0.7),
                    'alpha' : hp.uniform('alpha', 0.1, 0.5),
                    'scale_pos_weight' : 1,
                    'booster': 'gblinear',
                    'eval_metric': eval_metric,
                    'objective': objective,
                    'nthread' : 8,
                    'silent' : 1,
                    'seed': 7
                     }
            

        elif type=='tree':

            space = {
                    'booster': 'gbtree',
                    'eval_metric': eval_metric,
                    'objective': objective,
                    'eta' : hp.uniform('eta', 0.003, 0.3),
                    'colsample_bytree' : hp.uniform('colsample_bytree', 0.3, 0.95),
                    'subsample' : hp.uniform('subsample', 0.6, 0.95),
                    'max_depth' : hp.quniform('max_depth', 3, 15, 1),
                    'min_child_weight': hp.quniform('min_child_weight', 1, 100, 3),
                    'gamma': hp.loguniform('gamma', -3.0, 1.0),
                    'lambda': hp.loguniform('lambda', -3.0, 1.0),
                    'alpha': hp.uniform('alpha', 0.0, 1.0),
                    'scale_pos_weight' : 1,
                    'nthread' : 8,
                    'silent' : 1,
                    'seed': 7
                     }

        if stopping_rounds < 0:
            space['rounds'] = hp.quniform('rounds', 30, 700, 5)

        if task == 'multiclass_classification':
            space['num_class'] = n_classes

        best = fmin(score, space, algo = tpe.suggest, trials = trials, max_evals = int(total_evals / len(folds)))
        for key in space:
            if key in best:
                space[key] = best[key]
        
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