import pandas as pd
import numpy as np

np.random.seed(1)

import xgboost as xgb
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split, StratifiedKFold,StratifiedShuffleSplit,ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, r2_score
import pickle
import gc
from collections import Counter
from hyperopt import hp
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import log_loss,mean_squared_error
from scipy.optimize import fmin, fmin_powell,fmin_bfgs
from sklearn.feature_selection import SelectKBest, chi2
import os 

from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adadelta
from keras.layers.advanced_activations import LeakyReLU,PReLU,ELU

def get_hash_data_memory_efficient(train, test, colnames_to_hash, type = 'tfidf', verbose = True):
	"""
	Pandas dataframes train, test -> hashed values by columns 
	"""

	if type == 'tfidf':
		tfv = TfidfVectorizer(min_df=1)
	elif type == 'tfidf_modified':
		tfv = TfidfVectorizer(min_df=1, ngram_range = (1,1), sublinear_tf = True,binary = True, norm = 'l2')
	else:
		tfv = CountVectorizer(min_df = 1, binary = 1)

	c = 0
	for colname in colnames_to_hash:
	
		colname_to_hash = [colname]
		
		tfv.fit(train[colname].astype(np.str).fillna('Missing'))
		train_tf = tfv.transform(train[colname].astype(np.str).fillna('Missing'))
		test_tf = tfv.transform(test[colname].astype(np.str).fillna('Missing'))
		
		if c == 0:
			full_train_tf = train_tf
			full_test_tf = test_tf
		else:
			full_train_tf = sparse.hstack([full_train_tf, train_tf])
			full_test_tf = sparse.hstack([full_test_tf, test_tf])			

			if verbose:
				print (colname, full_train_tf.shape, full_test_tf.shape)

		c += 1

	return full_train_tf, full_test_tf

def get_hash_data_simple(train, test, colnames_to_hash, type = 'tfidf', verbose = True):
	"""
	Pandas dataframes train, test -> hashed values by columns 
	"""

	if type == 'tfidf':
		tfv = TfidfVectorizer(min_df=1)
	elif type == 'tfidf_modified':
		tfv = TfidfVectorizer(min_df=1, ngram_range=(1,1),sublinear_tf = True,binary = True, norm = 'l2')
	elif type == 'counter':
		tfv = CountVectorizer(min_df = 1, binary = 1)
	elif type == 'counter2':
		tfv = CountVectorizer(min_df = 1, binary = False)

	try:
		train = train[colnames_to_hash].astype(np.str).apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')
	except Exception:
		train = train[colnames_to_hash].astype('U').apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')
		
	if verbose:
		print ('-'*30,'df for hashing','\n',train)
	
	tfv.fit(train)

	try:
		test = test[colnames_to_hash].astype(np.str).apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')
	except Exception:
		test = test[colnames_to_hash].astype('U').apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')
	
	test = tfv.transform(test)
	
	if verbose:
		print ('-'*30,'df hashed','\n',train.shape)
		
	return train, test

def get_hash_data(train, test, colnames_to_hash, colnames_not_to_hash = [], type = 'tfidf', verbose = True):
	"""
	Pandas dataframes train, test -> hashed values by columns 
	"""

	df_all = pd.concat((train, test), axis=0, ignore_index=True)
	df = train

	split_len = len(train)

	if type == 'tfidf':
		tfv = TfidfVectorizer(min_df=1)
	elif type == 'tfidf_modified':
		tfv = TfidfVectorizer(min_df=1, ngram_range=(1,1),sublinear_tf = True,binary = True, norm = 'l2')
	else:
		tfv = CountVectorizer(min_df=1, binary=1)

	try:
		df = df[colnames_to_hash].astype(np.str).apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')
		df_all = df_all[colnames_to_hash].astype(np.str).apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')
	except Exception:
		df = df[colnames_to_hash].astype('U').apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')
		df_all = df_all[colnames_to_hash].astype('U').apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')

	if verbose:
		print ('-'*30,'df for hashing','\n',df)
	
	tfv.fit(df)
	
	del df
	gc.collect()
	
	df_tfv = tfv.transform(df_all)
	
	if verbose:
		print ('-'*30,'df hashed','\n',df_tfv.shape)
	
	not_hashed_df_train = train[colnames_not_to_hash]
	not_hashed_df_test = test[colnames_not_to_hash]
	
	not_hashed_df_train = sparse.coo_matrix(not_hashed_df_train.values)
	not_hashed_df_test = sparse.coo_matrix(not_hashed_df_test.values)

	train = df_tfv[:split_len, :]
	test = df_tfv[split_len:, :]
	
	if len(colnames_not_to_hash) > 0:
		train = sparse.hstack([not_hashed_df_train, train])
		test = sparse.hstack([not_hashed_df_test, test])

	del df_all
	del df_tfv
	gc.collect()
	
	return train, test


def find_params_xgb(train, Y, type = 'linear', objective='', eval_metric = '', n_classes = 2, folds = 5, total_evals = 50, sparse = True, scale_pos_weight = 1):
	"""
	train dataset, labels, type -> dict of optimal params
	"""

	skf = StratifiedKFold(Y, n_folds=folds, random_state = 777)

	def score(params):
		CVs = 0
		CV_score = 0.0

		for train_index, val_index in skf:

			if CVs == 0:
	
				print ("Training with params : ")
				print (params)
			
			if sparse:
	
				X_train = train.tocsr()[train_index,:].tocoo()
				X_val = train.tocsr()[val_index,:].tocoo()
				y_train = Y[train_index]
				y_val = Y[val_index]

				X_train = X_train.tocsc()[:, :X_val.shape[1]].tocoo()
				
				print X_train.shape, X_val.shape, y_train.shape, y_val.shape

				dtrain = xgb.DMatrix(X_train.tocsc(), y_train, missing = np.nan)
				dvalid = xgb.DMatrix(X_val.tocsc(), y_val, missing = np.nan)
	
			else:
	
				X_train = train[train_index,:]
				X_val = train[val_index,:]
				y_train = Y[train_index]
				y_val = Y[val_index]

				X_train = X_train[:, :X_val.shape[1]]
			
				dtrain = xgb.DMatrix(X_train, y_train, missing = np.nan)
				dvalid = xgb.DMatrix(X_val, y_val, missing = np.nan)
			

			watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
			if 'max_depth' in params:
				params['max_depth'] = int(params['max_depth'])
			model = xgb.train(params, dtrain, 1000, evals=watchlist, verbose_eval=True, early_stopping_rounds = 1)
			score = model.best_score
			
			CVs += 1
			CV_score += score

			print ('FOLD', CVs, 'SCORE', score)

		CV_score /= float(CVs)
		print ("\tCV_score {0}\n\n".format(CV_score))

		if params['eval_metric'] != 'auc':
			return {'loss': CV_score, 'status': STATUS_OK}
		else:
			print -CV_score
			return {'loss': -CV_score, 'status': STATUS_OK}
			
	def optimize(trials):
		
		eval_metric_ = eval_metric
		objective_ = objective

		if type == 'linear':
			
			if eval_metric == '':
				eval_metric_ = 'mlogloss'

			if objective == '':
				objective_ = 'multi:softprob'

			space = {
					'eta': hp.quniform('eta', 0.003, 0.1, 0.01),
					'lambda' : hp.quniform('lambda', 0.1, 0.7, 0.15),
					'lambda_bias' : hp.quniform('lambda_bias', 0.1, 0.7, 0.15),
					'alpha' : hp.quniform('alpha', 0.1, 0.5, 0.1),
					'scale_pos_weight' : scale_pos_weight,
					'booster': 'gblinear',
					'num_class' : n_classes,
					'eval_metric': eval_metric_,
					'objective': objective_,
					'nthread' : 8,
					'silent' : 0,
					'seed': 7
					 }

		elif type == 'linear_softmax':
			
			if eval_metric == '':
				eval_metric_ = 'mlogloss'
			if objective == '':
				objective_ = 'multi:softmax'

			space = {
					'eta': hp.quniform('eta', 0.003, 0.1, 0.01),
					'lambda' : hp.quniform('lambda', 0.1, 0.7, 0.15),
					'lambda_bias' : hp.quniform('lambda_bias', 0.1, 0.7, 0.15),
					'alpha' : hp.quniform('alpha', 0.1, 0.5, 0.1),
					'scale_pos_weight' : scale_pos_weight,
					'booster': 'gblinear',
					'num_class' : n_classes,
					'eval_metric': eval_metric_,
					'objective': objective_,
					'nthread' : 8,
					'silent' : 0,
					'seed': 7
					 }

		elif type == 'linear_reg':
			
			if eval_metric == '':
				eval_metric_ = 'rmse'
			if objective == '':
				objective_ = 'reg:linear'

			space = {
					'eta': hp.quniform('eta', 0.003, 0.1, 0.01),
					'lambda' : hp.quniform('lambda', 0.1, 0.7, 0.15),
					'lambda_bias' : hp.quniform('lambda_bias', 0.1, 0.7, 0.15),
					'alpha' : hp.quniform('alpha', 0.1, 0.5, 0.1),
					'scale_pos_weight' : scale_pos_weight,
					'booster': 'gblinear',
					'eval_metric': eval_metric_,
					'objective': objective_,
					'nthread' : 8,
					'silent' : 0,
					'seed': 7
					 }

		elif type=='tree':

			if eval_metric == '':
				eval_metric_ = 'mlogloss'
			if objective == '':
				objective_ = 'multi:softprob'

			space = {
					'eta' : hp.quniform('eta', 0.003, 0.1, 0.01),
					'colsample_bytree' : hp.quniform('colsample_bytree', 0.3, 0.9, 0.1),
					'subsample' : hp.quniform('subsample', 0.6, 0.9, 0.1),
					'max_depth' : hp.quniform('max_depth', 3, 9, 1),
					'scale_pos_weight' : scale_pos_weight,
					'booster': 'gbtree',
					'num_class' : n_classes,
					'eval_metric': eval_metric_,
					'objective': objective_,
					'nthread' : 8,
					'silent' : 1,
					'seed': 7
					 }

		elif type=='tree_softmax':

			if eval_metric == '':
				eval_metric_ = 'mlogloss'
			if objective == '':
				objective_ = 'multi:softmax'

			space = {
					'eta' : hp.quniform('eta', 0.03, 0.1, 0.01),
					'colsample_bytree' : hp.quniform('colsample_bytree', 0.3, 0.9, 0.1),
					'subsample' : hp.quniform('subsample', 0.6, 0.9, 0.1),
					'max_depth' : hp.quniform('max_depth', 3, 9, 1),
					'scale_pos_weight' : scale_pos_weight,
					'booster': 'gbtree',
					'num_class' : n_classes,
					'eval_metric': eval_metric_,
					'objective': objective_,
					'nthread' : 8,
					'silent' : 1,
					'seed': 7
					 }

		elif type=='tree_binary':

			if eval_metric == '':
				eval_metric_ = 'auc'
			if objective == '':
				objective_ = 'binary:logistic'

			space = {
					'eta' : hp.quniform('eta', 0.003, 0.5, 0.01),
					'colsample_bytree' : hp.quniform('colsample_bytree', 0.3, 0.9, 0.1),
					'subsample' : hp.quniform('subsample', 0.6, 0.9, 0.1),
					'max_depth' : hp.quniform('max_depth', 2, 12, 1),
					'scale_pos_weight' : scale_pos_weight,
					'booster': 'gbtree',
					'eval_metric': eval_metric_,
					'objective': objective_,
					'nthread' : 8,
					'silent' : 1,
					'seed': 7
					 }

		elif type=='tree_reg':

			if eval_metric == '':
				eval_metric_ = 'rmse'
			if objective == '':
				objective_ = 'reg:linear'

			space = {
					'eta' : hp.quniform('eta', 0.003, 0.1, 0.01),
					'colsample_bytree' : hp.quniform('colsample_bytree', 0.3, 0.9, 0.1),
					'subsample' : hp.quniform('subsample', 0.6, 0.9, 0.1),
					'max_depth' : hp.quniform('max_depth', 2, 8, 1),
					'scale_pos_weight' : scale_pos_weight,
					'booster': 'gbtree',
					'eval_metric': eval_metric_,
					'objective': objective_,
					'nthread' : 8,
					'silent' : 1,
					'seed': 7
					 }


		best = hyperopt.fmin(score, space, algo=tpe.suggest, trials=trials, max_evals = int(total_evals/folds))
		for key in space:
			if key in best:
				space[key] = best[key]
		
		print ('-'*50,'BEST PARAMS',space)
		return space

	trials = Trials()
	best = optimize(trials)
	
	return best

def train_xgb(train, Y, params, type='multiclass_classification',train_size = 0.9, seed = 1, sparse = False, verbose_eval = True):
	"""
	dataset, params -> fitted model and its validation score
	"""
	if type=='multiclass_classification' or type=='binary_classification':
		skf =  StratifiedShuffleSplit(Y, train_size=train_size, test_size = int((1.0-train_size)*train.shape[0])-1, random_state=seed)
	else:
		skf =  ShuffleSplit(Y.shape[0], train_size=train_size, test_size = None, random_state=seed)
		
	for train_index, val_index in skf:
		break

	if sparse:
	
		X_train, X_val, y_train, y_val = train.tocsr()[train_index,:].tocoo(), train.tocsr()[val_index,:].tocoo(), Y[train_index], Y[val_index]
		dtrain = xgb.DMatrix(X_train.tocsc(), y_train, missing = np.nan)
		dvalid = xgb.DMatrix(X_val.tocsc(), y_val, missing = np.nan)
	
	else:
	
		X_train, X_val, y_train, y_val = train[train_index,:], train[val_index,:], Y[train_index], Y[val_index]
		dtrain = xgb.DMatrix(X_train, y_train, missing = np.nan)
		dvalid = xgb.DMatrix(X_val, y_val, missing = np.nan)
	
	watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
	
	if 'max_depth' in params:
		params['max_depth'] = int(params['max_depth'])

	print ("Training with params : ")
	print (params)
	
	model = xgb.train(params, dtrain, 2000, evals=watchlist, verbose_eval=verbose_eval, early_stopping_rounds = 1)
	score = model.best_score

	return model, score

def predict_xgb(X, model, sparse = False, proba = True):
	"""
	features, fitted model -> predictions
	"""
	if sparse:
		if proba:
			preds = model.predict_proba(xgb.DMatrix(X.tocsc(), missing = np.nan))
		else:
			preds = model.predict(xgb.DMatrix(X.tocsc(), missing = np.nan))
	else:
		if proba:
			preds = model.predict_proba(xgb.DMatrix(X, missing = np.nan))
		else:
			preds = model.predict(xgb.DMatrix(X, missing = np.nan))
			

	return preds

def predict_classes_xgb(X, model, sparse = False):
	"""
	features, fitted model -> predictions
	"""
	if sparse:
		preds = model.predict(xgb.DMatrix(X.tocsc(), missing = np.nan))
	else:
		preds = model.predict(xgb.DMatrix(X, missing = np.nan))
	

	return preds

def CV_evaluation(X, Y, model, type='classification', folds = 5, sparse = False):
	"""
	dataset, fitted model, task , nfolds -> Cross-Validation score
	"""
	CVs = 0
	CV_score = 0.0

	skf = StratifiedKFold(Y, n_folds=folds, random_state = 777)

	for train_index, val_index in skf:

		if CVs == 0:
			print ("Training  model : ")
			print (model)
		
		if sparse:
			X_train = X.tocsr()[train_index,:].tocoo()
			X_val = X.tocsr()[val_index,:].tocoo()
		else:
			X_train = X[train_index,:]
			X_val = X[val_index,:]
			
		y_train = Y[train_index]
		y_val = Y[val_index]

		model.fit(X_train, y_train)

		if type=='classification':
			preds = model.predict_proba(X_val)
			score = log_loss(y_val, preds)
		else:
			preds = model.predict(X_val)
			score = mean_squared_error(y_val, preds)
			
		CVs += 1
		CV_score += score

		print ('FOLD', CVs, 'SCORE', score)

	CV_score /= float(CVs)
	print ("\tCV_score {0}\n\n".format(CV_score))
	return CV_score

def create_features_for_stack(X, Y, test_X, model, type='multiclass_classification', features_to_keep = 'all', nfolds = 5, sparse_matrix = True, seed = 1, n_classes=None):
	"""
	train dataset, test features, model, task type -> datasets of stacked model predictions
	"""
	skf = StratifiedKFold(Y, n_folds=nfolds, random_state = seed)
	if type == 'multiclass_classification':
		preds = np.zeros((X.shape[0],n_classes), dtype = np.float32)
	elif type == 'binary_classification':
		preds = np.zeros((X.shape[0],1), dtype = np.float32)
	elif type=='regression':
		preds = np.zeros((X.shape[0],1), dtype = np.float32)
		
	if features_to_keep == 'all':
		new_X_train = X
		new_X_test = test_X
	elif features_to_keep > 0:
		new_X_train = X[:, :features_to_keep]
		new_X_test = test_X[:, :features_to_keep]
	else:
		new_X_train = None
		new_X_test = None

	CVs = 0
	CV_score = 0.0

	for train_index, val_index in skf:

		if CVs == 0:
			print ("Training  model : ")
			print (model)
		CVs += 1

		if sparse_matrix:
			X_train = X.tocsr()[train_index,:].tocoo()
			X_val = X.tocsr()[val_index,:].tocoo()
		else:
			X_train = X[train_index,:]
			X_val = X[val_index,:]
			
		y_train = Y[train_index]
		y_val = Y[val_index]

		if isinstance(model, dict) and 'booster' in model:
		
			model_fitted, score = train_xgb(X_train, y_train, model, type=type, train_size = 0.95, seed = seed, sparse = sparse_matrix,verbose_eval=False )
			cur_preds = predict_xgb(X_val, model_fitted, sparse = sparse_matrix)
		
		elif isinstance(model, dict):
		
			cur_preds = train_predict_keras(X_train, y_train, X_val, model, type=type)
			print(cur_preds)
		
		else:
		
			model.fit(X_train, y_train)
			cur_preds = model.predict_proba(X_val)
		
		if type == 'multiclass_classification':
		
			preds[val_index, :] = cur_preds
			score = log_loss(y_val, cur_preds)
		
		elif type == 'binary_classification':
		
			preds[val_index, 0] = cur_preds[:,0]
			print(preds[val_index])
			score = log_loss(y_val, cur_preds)
		
		elif type=='regression':
		
			cur_preds = cur_preds.reshape(cur_preds.shape[0],)
			preds[val_index,0] = cur_preds
			score = mean_squared_error(y_val, cur_preds)

		CV_score += score
		print ('FOLD', CVs, 'SCORE', score)
	
	
	CV_score /= float(CVs)
	print ("\tCV_score {0}\n\n".format(CV_score))
	
	if isinstance(model, dict) and 'booster' in model:
	
		model_fitted, score = train_xgb(X, Y, model, type=type,train_size = 0.95, seed = seed, sparse = sparse_matrix,verbose_eval=False )
		preds_test = predict_xgb(test_X, model_fitted, sparse = sparse_matrix)
		if type == 'regression':
			preds_test = preds_test.reshape(preds_test.shape[0],1)
			print('test shape',preds_test.shape)
		
	elif isinstance(model, dict):
	
		preds_test = train_predict_keras(X, Y, test_X, model, type=type)
		if type=='binary_classification' or type=='regression':
			preds_test = preds_test[:,0]
			preds_test = preds_test.reshape(preds_test.shape[0],1)
		print(preds_test)
	
	else:
	
		model.fit(X, Y)
		preds_test = model.predict_proba(test_X)
	
	if new_X_train != None:
	
		if sparse_matrix:
			new_X_train = sparse.hstack([new_X_train, sparse.coo_matrix(preds)])
			new_X_test = sparse.hstack([new_X_test, sparse.coo_matrix(preds_test)])
		
		else:
			new_X_train = np.hstack(new_X_train, preds)
			new_X_test = np.hstack(new_X_test, preds_test)
	
	else:
		if sparse_matrix:
			new_X_train = sparse.coo_matrix(preds)
			new_X_test = sparse.coo_matrix(preds_test)
		
		else:
			new_X_train = preds
			new_X_test = preds_test

	return new_X_train, new_X_test, CV_score

def create_stack(X, Y, test_X, models, type='multiclass_classification', n_classes = None, features_to_keep = 0, nfolds = 5, sparse_matrix = True):
	"""
	train dataset, test features, model, task type -> datasets of stacked model predictions
	"""
	m = 0
	for model in models:
		if m == 0:
		
			stack_X_train, stack_X_test, _ = create_features_for_stack(X, Y, test_X, model, type=type, features_to_keep = features_to_keep, nfolds = nfolds, sparse_matrix = sparse_matrix, seed = m,n_classes = n_classes)
		else:
		
			cur_stack_X_train, cur_stack_X_test, _ = create_features_for_stack(X, Y, test_X, model, type=type, features_to_keep = 0, nfolds = nfolds, sparse_matrix = sparse_matrix, seed = m,n_classes = n_classes)
			if sparse_matrix:
		
				stack_X_train = sparse.hstack([stack_X_train, sparse.coo_matrix(cur_stack_X_train)])
				stack_X_test = sparse.hstack([stack_X_test, sparse.coo_matrix(cur_stack_X_test)])
		
			else:
				stack_X_train = np.hstack([stack_X_train, cur_stack_X_train])
				stack_X_test = np.hstack([stack_X_test, cur_stack_X_test])

		print('X for stacking shape', stack_X_train.shape, stack_X_test.shape)
		m += 1

	return stack_X_train, stack_X_test

def make_mean_of_models(models, train_X, Y, test, sparse = False):
	"""
	models, dataset, test features -> mean predictions of models
	"""
	nmodels = len(models)
	n = 0
	for model in models:
	
		print ('training model', model)

		if isinstance(model, dict):#XGB
	
			model_fitted, score = train_xgb(train_X, Y, model, train_size = 0.9, seed = n, sparse = sparse, verbose_eval = True)
			cur_preds = predict_xgb(test, model_fitted, sparse = sparse)
	
		else:
	
			model.fit(train_X, Y)
			cur_preds = model.predict_proba(test)

		if n == 0:
			preds = cur_preds
		else:
			preds += cur_preds
		n += 1

	preds /= float(n)
	return preds

def batch_generator(X, y, batch_size, shuffle):

	number_of_batches = X.shape[0]//batch_size
	counter = 0
	sample_index = np.arange(X.shape[0])

	if shuffle:

		np.random.shuffle(sample_index)

	while True:

		batch_index = sample_index[batch_size*counter:min(X.shape[0],batch_size*(counter+1))]
		X_batch = X[batch_index,:].toarray()
		y_batch = y[batch_index]
		counter += 1
			
		yield X_batch, y_batch
		if (counter == number_of_batches):
			if shuffle:
				np.random.shuffle(sample_index)
			counter = 0

def generate_model(params,type='multiclass_classification', n_classes = None):
	"""
	params -> Keras model
	"""
	hidden_sizes, input_dim, dropouts, init = params['hidden_sizes'], params['input_dim'], params['dropouts'], params['init']
	print (hidden_sizes, input_dim, dropouts, init)

	n_layers = len(hidden_sizes)

	model = Sequential()

	for l in range(n_layers):
		if l == 0:
			model.add(Dense(hidden_sizes[l], input_dim=input_dim, init=init))
		else:		
			model.add(Dense(hidden_sizes[l], init=init))
		
		if 'activation' in params:
			if params['activation']=='prelu':
				model.add(Activation(PReLU()))
			elif params['activation']=='relu':
				model.add(Activation('relu'))
		
		if 'dropouts' in params:
			model.add(Dropout(dropouts[l]))

	if type=='multiclass_classification':
	
		model.add(Dense(n_classes, init=init))
		model.add(Activation('softmax'))
	
	elif type=='binary_classification':
	
		model.add(Dense(2, init=init))
		model.add(Activation('softmax'))
	
	elif type=='regression':
	
		model.add(Dense(1, init=init))
			

	optimizer_adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

	if type=='multiclass_classification':
		loss_ = 'categorical_crossentropy'
	elif type=='binary_classification':
		loss_ = 'binary_crossentropy'
	elif type=='regression':
		loss_ = 'mse'

	model.compile(loss=loss_, optimizer=optimizer_adam)
	return model

def CV_evaluation_keras(train_X, Y, params, type='multiclass_classification'):
	"""
	datasets, model params -> Cross-Validation score
	"""
	np.random.seed(7)

	skf = StratifiedKFold(Y, n_folds=5, random_state = 1)
	batch_size = 256
	nb_epoch = 2
	
	CV_score = 0.0
	CVs = 1
	
	for train_index, val_index in skf:
		
		cur_train = train_X.tocsr()[train_index,:]
		cur_val = train_X.tocsr()[val_index,:]
		
		Y_train = np_utils.to_categorical(Y[train_index])
		Y_val = np_utils.to_categorical(Y[val_index])
		
		val_size = batch_size*(cur_val.shape[0]//batch_size)
		cur_val = cur_val[:val_size]
		Y_val = Y_val[:val_size]
		
		batch_size = params['batch_size']
		nb_epoch = params['nb_epoch']

		model = generate_model(params,type=type)
		model.fit_generator(generator=batch_generator(cur_train, Y_train, batch_size, True),
			nb_epoch=nb_epoch, 
			samples_per_epoch=cur_train.shape[0])
		probas = model.predict_generator(generator=batch_generator(cur_val, Y_val, batch_size, False), val_samples = cur_val.shape[0])
		
		if 'classification' in type:
			score = log_loss(Y[val_index][:val_size], probas)
		else:
			score = mean_squared_error(Y[val_index][:val_size], probas)
		
		CV_score += score
		print ('FOLD', CVs, 'SCORE', score)
		CVs += 1

	CV_score /= float(CVs-1)
	print ("\tCV_score {0}\n\n".format(CV_score))

	return CV_score

def train_predict_keras(train_X, Y, test, params, type='multiclass_classification'):
	"""
	train dataset, test features, model params -> model and predicitons
	"""
	train_X = train_X.tocsr()
	test = test.tocsr()

	np.random.seed(params['seed'])

	if 'classification' in type:
		print(Y,Y.shape)
		min_class = np.min(Y)
		Y -= min_class
		n_classes = np.max(Y) + 1
		print (min_class,Y, 'n_classes',n_classes)

	else:
		n_classes = 0

	model = generate_model(params, type=type, n_classes = n_classes)
	
	if 'classification' in type:
		fit_y =  np_utils.to_categorical(Y)
	else:
		fit_y  = Y

	model.fit_generator(generator=batch_generator(train_X, fit_y, batch_size=params['batch_size'], shuffle=True),
			nb_epoch=params['nb_epoch'], 
			samples_per_epoch=train_X.shape[0])

	if 'classification' in type:
		probas = model.predict_generator(generator=batch_generator(test, np_utils.to_categorical(np.zeros((test.shape[0],),dtype=np.int32)), 1, shuffle=False), val_samples = test.shape[0])
	else:
		probas = model.predict_generator(generator=batch_generator(test, np.zeros((test.shape[0],),dtype=np.int32), 1, shuffle=False), val_samples = test.shape[0])
	
	print ('keras preds', probas.shape)
	
	return probas
