'''
dataset is taken from https://archive.ics.uci.edu/ml/datasets/Statlog+%28Australian+Credit+Approval%29
'''

import kagglelib
from kagglelib.feature_engineering import *
from kagglelib.preprocessing import *
from kagglelib.model_tuning import *
from kagglelib.model_training import *
from kagglelib.model_predicting import *
from kagglelib.utils import *
from kagglelib.Model import Model
from kagglelib.utils import generate_keras_model
from kagglelib.feature_selection import genetic_feature_selection
from kagglelib.preprocessing import *
from kagglelib.model_ensembles import *
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import os

#read data and name the columns
dirpath = os.path.dirname(os.path.abspath(__file__)) + '/datasets'
data = pd.read_csv(dirpath + '/australian.dat', header = None, delimiter = ' ')
data.columns = ['f%d' % i for i in xrange(data.shape[1])]

#divide dataset into two equal parts: train and test, target variable is f14
train = data.ix[:data.shape[0] / 2]
train_y = train['f14'].values
train.drop('f14', inplace = True, axis = 1)
test = data.ix[data.shape[0] / 2 :]
test_y = test['f14'].values
test.drop('f14', inplace = True, axis = 1)
print train.head()

#create 3-folds cross-validation split
folds = make_folds(train_y, n_folds = 3, random_seed = 7, type = 'stratified')

#specify categorical and numerical columns 
categorical_columns = ['f0', 'f3', 'f4', 'f5', 'f7', 'f8', 'f10', 'f11']
numerical_columns = ['f1', 'f2', 'f6', 'f9', 'f12']

#create categorical interaction between f0 and f3 columns
categorical_columns += make_categorical_interactions(train, test, ['f0', 'f3'], order = 2)

#create numerical multiplication interaction between f1 and f2 columns
numerical_columns += make_numerical_interactions(train, test, ['f1', 'f2'], 'mult', order = 2)

#create additional columns with logarithm transformation of f2 and f6 columns
numerical_columns += logarithm(train, test, ['f2', 'f6'])
#create additional columns with trigonometrical transformation of f1 column
numerical_columns += trigonometry(train, test, ['f1'])

#cast categorical columns to numerical values
for col in categorical_columns:
	train.loc[:, col] = train[col].astype(int)
	test.loc[:, col] = test[col].astype(int)

#check current dataset
print train.head()
print numerical_columns
print categorical_columns

#make categorical target encoding for all categorical columns
categorical_target_encoding(train, train_y, test, categorical_columns, folds, simple_scheme = False, inner_type='random', threshold = 1, alpha = 10, verbose = False)

#select features with genetic algorithm with xgbosot model with some set parameters, metric is AUC
model_xgb = Model('xgboost')
model_xgb.params = {'booster': 'gbtree','eval_metric': 'logloss','objective': 'binary:logistic','eta' : 0.05,'colsample_bytree' : 0.9,'max_depth' : 7, 'rounds' : 50, 'nthread' : 8,'seed': 7, 'silent': 1}
selected_features, best_score = genetic_feature_selection(train, train_y, 'binary_classification', model_xgb, folds, 'auc', iterations = 10, generation_size = 15, generation_best_ratio = 5, verbose = False)

#see which features were selected and keep only them in both train and test 
print selected_features, best_score
train = train.loc[:, selected_features]
test = test.loc[:, selected_features]

#normalize data to train neural nets
train_normed, test_normed = normalize_data(train.values, test.values, max_norm = True)

#create neural net model
model_keras = Model('keras')
model_keras.params = {'input_dim' : train.shape[1], 'hidden_sizes' : [10, 10], 'hidden_activation': 'tanh', 'dropouts' : [0.1, 0.1], 'init' : 'glorot_normal', 'output_activation' : 'sigmoid'}
#check cv score with averaging several epoches and without
model_keras.cv_score = CV_score_keras(train_normed, train_y, 'binary_classification', model_keras.params, 'auc', folds, stopping_rounds = 1, verbose = False, average_epochs = -1)
print model_keras, '\n'
model_keras.cv_score = CV_score_keras(train_normed, train_y, 'binary_classification', model_keras.params, 'auc', folds, stopping_rounds = 1, verbose = False, average_epochs = 5)
print model_keras, '\n'
#train and check score on test with averaging and without
model_keras.model, _, test_preds_nnet = train_keras(train_normed, train_y, test_normed, 'binary_classification', model_keras.params, 'logloss', epochs = 20, train_ratio = 0.9, stopping_rounds = 1, average_epochs = -1, verbose = False)
model_keras.model, _, test_preds_nnet_avg = train_keras(train_normed, train_y, test_normed, 'binary_classification', model_keras.params, 'logloss', epochs = 20, train_ratio = 0.9, stopping_rounds = 1, average_epochs = 10, verbose = False)
print roc_auc_score(test_y, test_preds_nnet), ' | with avg epochs: ', roc_auc_score(test_y, test_preds_nnet_avg) 

#create xgboost model
model_xgb = Model('xgboost')
#find best parameters with hyperopt
model_xgb.params, best_xgb_score = find_params_xgb(train.values, train_y, 'binary_classification', 'tree', 'auc', 2, folds, total_evals = 300, stopping_rounds = -1, missing = np.nan, verbose = False)
#calc cv score with best parameters
model_xgb.cv_score = CV_score_xgb(train.values, train_y, model_xgb.params, 'auc', folds, sparse = False, stopping_rounds = -1, missing = np.nan, verbose = False)
print model_xgb, '\n'
#train xgboost model
model_xgb.model, _ = train_xgb(train.values, train_y, 'binary_classification', model_xgb.params, 'auc', sparse = False, train_ratio = 0.8, stopping_rounds = -1, missing = np.nan, verbose = False)
#make predictions on test
test_preds_xgb = predict_xgb(test.values, model_xgb.model)

#create lightgbm model
model_lgbm = Model('lightgbm')
#find best parameters with hyperopt
model_lgbm.params, best_lgbm_score = find_params_lgbm(train, train_y, 'binary_classification', categorical_columns, 'auc', 2, folds, total_evals = 300, stopping_rounds = -1, verbose = False)
#calc cv score with best parameters
model_lgbm.cv_score = CV_score_lgbm(train, train_y, model_lgbm.params, categorical_columns, 'auc', folds, stopping_rounds = -1, verbose = False)
print model_lgbm, '\n'
#train lightgbm model
model_lgbm.model, _ = train_lgbm(train, train_y, 'binary_classification', model_lgbm.params, categorical_columns, 'auc', train_ratio = 0.8, stopping_rounds = -1, verbose = False)
#make predictions on test
test_preds_lgbm = predict_lgbm(test.values, model_lgbm.model)

#create sklearn model
from sklearn.ensemble import RandomForestClassifier
model_sklearn = Model('sklearn')
model_sklearn.model = RandomForestClassifier()
model_sklearn.params = {'max_depth': 10, 'n_estimators' : 50, 'n_jobs' : 8} 
#calc cv score
model_sklearn.cv_score = CV_score_sklearn(train, train_y, model_sklearn, 'binary_classification', 'auc', folds, verbose = True)
print model_sklearn, '\n'
#train sklearn model
model_sklearn.model = train_sklearn(train.values, train_y, model_sklearn)
#make predictions on test
test_preds_sklearn = predict_sklearn(test.values, model_sklearn.model, 'binary_classification', 'auc')

print 'neural net test score : %f' % roc_auc_score(test_y, test_preds_nnet_avg) 
print 'xgboost test score : %f' % roc_auc_score(test_y, test_preds_xgb) 
print 'lightgbm test score : %f' % roc_auc_score(test_y, test_preds_lgbm) 
print 'sklearn random forest test score : %f' % roc_auc_score(test_y, test_preds_sklearn) 

#create stacking on our 4 models (out-of-fold method)
train_stack, test_stack = create_stack(train_normed, train_y, test_normed, [model_keras, model_xgb, model_lgbm, model_sklearn], folds, task = 'binary_classification', eval_metric = 'auc', categorical_features = categorical_columns, train_ratio = 0.8, n_classes = 2, sparse = False, stopping_rounds = [-1,-1,-1,-1], epochs = 100, average_epochs = 5, random_seed = 1, verbose = False)
print train_stack, test_stack

#let's build a model on our stackingm it will be linear xgboost
#create xgboost model
model_xgb2 = Model('xgboost')
#find best parameters with hyperopt
model_xgb2.params, best_xgb_score = find_params_xgb(train_stack, train_y, 'binary_classification', 'linear', 'auc', 2, folds, total_evals = 300, stopping_rounds = -1, missing = np.nan, verbose = False)
#calc cv score with best parameters
model_xgb2.cv_score = CV_score_xgb(train_stack, train_y, model_xgb2.params, 'auc', folds, sparse = False, stopping_rounds = -1, missing = np.nan, verbose = False)
print model_xgb2, '\n'
#train xgboost model
model_xgb2.model, _ = train_xgb(train_stack, train_y, 'binary_classification', model_xgb2.params, 'auc', sparse = False, train_ratio = 0.8, stopping_rounds = -1, missing = np.nan, verbose = False)
#make predictions on test and ensure that single model on stacking has surpassed scores of all single models
test_preds_xgb2 = predict_xgb(test_stack, model_xgb2.model)
print 'xgboost test stacking score : %f' % roc_auc_score(test_y, test_preds_xgb2) 


