
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
from sklearn.metrics import log_loss
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

def merge_submissions(subms_dir):
	"""

	"""
	files = os.listdir(subms_dir)
	f = open('merged_subm.csv','w')
	csv_readers = [csv.reader(file) for file in files]
	while 1:
		if row == 0:
			f.write(row)
	f.close()

def create_mean_features(train, Y, test_dataframes):
	skf = StratifiedKFold(Y, n_folds=5)

	train['mean_age_brand'] = 0
	train['mean_gender_brand'] = 0
	train['mean_age_model'] = 0
	train['mean_gender_model'] = 0


	alpha = 10

	brand_group_names = ['brand_group_prob' + str(i) for i in range(12)]
	model_group_names = ['model_group_prob' + str(i) for i in range(12)]
	
	for col in brand_group_names + model_group_names:
		for i in range(len(test_dataframes)):
			test_dataframes[i][col] = 0

			# test_dataframes[i]['mean_age_brand'] = 0
			# test_dataframes[i]['mean_gender_brand'] = 0
			# test_dataframes[i]['mean_age_model'] = 0
			# test_dataframes[i]['mean_gender_model'] = 0

		train[col] = 0

	for train_index, val_index in skf:
		#print (train_index, val_index)
		cur_train = train.iloc[train_index]
		
		global_mean_age = cur_train['age'].mean()
		global_mean_gender = cur_train['gender'].mean()

		# mean_age_brand = cur_train.groupby('phone_brand')['age'].mean()
		# mean_gender_brand = cur_train.groupby('phone_brand')['gender'].mean()
		# mean_age_model = cur_train.groupby('device_model')['age'].mean()
		# mean_gender_model = cur_train.groupby('device_model')['gender'].mean()
			
		#print (mean_age_brand.keys)

		#mean_group_brands = []

		for group in range(12):
			ind = np.where(Y[train_index] == group)[0]
			overall_prob = float(ind.shape[0]) / Y.shape[0]
			
			brand_counter = cur_train.iloc[ind].groupby('phone_brand')['phone_brand'].count()
			model_counter = cur_train.iloc[ind].groupby('device_model')['device_model'].count()

			brands = set(brand_counter.keys())
			models = set(model_counter.keys())

			brand_total_counter = cur_train.groupby('phone_brand')['phone_brand'].count()
			model_total_counter = cur_train.groupby('device_model')['device_model'].count()

			#print(brand_counter)
			#print(model_counter)

			#mean_group_brand = brand_counter.div(cur_train.groupby('phone_brand')['phone_brand'].count(), axis = 0, fill_value = 0.0)
			#mean_group_model = model_counter.div(cur_train.groupby('device_model')['device_model'].count(), axis = 0, fill_value = 0.0)
			
			#print(overall_prob)
			#print(brand_counter + 1)

			#mean_group_brand = (mean_group_brand.mul(brand_counter, axis = 0, fill_value = 0.0) + overall_prob).div(brand_counter + 1, axis = 0, fill_value = 0.0)
			#print(mean_group_brand)
			col_name = brand_group_names[group]
			train[col_name][val_index] = train.iloc[val_index]['phone_brand'].map(lambda x: ((brand_counter[x] + overall_prob*alpha) / (brand_total_counter[x] + alpha) if x in brands else 0.0))
			
			col_name = model_group_names[group]
			train[col_name][val_index] = train.iloc[val_index]['device_model'].map(lambda x: ((model_counter[x]  + overall_prob*alpha) / (model_total_counter[x] + alpha) if x in models else 0.0))
			
		# train['mean_age_brand'][val_index] = train.iloc[val_index]['phone_brand'].map(lambda x: mean_age_brand[x] if x in brands else global_mean_age) / 100.0 
		# train['mean_gender_brand'][val_index] = train.iloc[val_index]['phone_brand'].map(lambda x: mean_gender_brand[x] if x in brands else global_mean_gender)
		# train['mean_age_model'][val_index] = train.iloc[val_index]['device_model'].map(lambda x: mean_age_model[x] if x in models else global_mean_age) / 100.0
		# train['mean_gender_model'][val_index] = train.iloc[val_index]['device_model'].map(lambda x: mean_gender_model[x] if x in models else global_mean_gender)
		
		#print (train.iloc[val_index])


	cur_train = train
	
	global_mean_age = cur_train['age'].mean()
	global_mean_gender = cur_train['gender'].mean()

	# mean_age_brand = cur_train.groupby('phone_brand')['age'].mean()
	# mean_gender_brand = cur_train.groupby('phone_brand')['gender'].mean()
	# mean_age_model = cur_train.groupby('device_model')['age'].mean()
	# mean_gender_model = cur_train.groupby('device_model')['gender'].mean()
		
	#print (mean_age_brand.keys)
	#brands = set(mean_age_brand.keys())
	#models = set(mean_age_model.keys())

	for group in range(12):
		ind = np.where(Y == group)[0]
		overall_prob = float(ind.shape[0]) / Y.shape[0]

		brand_counter = cur_train.iloc[ind].groupby('phone_brand')['phone_brand'].count()
		model_counter = cur_train.iloc[ind].groupby('device_model')['device_model'].count()

		brands = set(brand_counter.keys())
		models = set(model_counter.keys())

		brand_total_counter = cur_train.groupby('phone_brand')['phone_brand'].count()
		model_total_counter = cur_train.groupby('device_model')['device_model'].count()

		#mean_group_brand = cur_train.iloc[ind].groupby('phone_brand')['phone_brand'].count().div(cur_train.groupby('phone_brand')['phone_brand'].count(), axis = 0, fill_value = 0.0)
		#mean_group_model = cur_train.iloc[ind].groupby('device_model')['device_model'].count().div(cur_train.groupby('device_model')['device_model'].count(), axis = 0, fill_value = 0.0)
	   
		for i in range(len(test_dataframes)):
			col_name = brand_group_names[group]
			test_dataframes[i][col_name] = test_dataframes[i]['phone_brand'].map(lambda x:  ((brand_counter[x]  + overall_prob*alpha) / (brand_total_counter[x] + alpha) if x in brands else 0.0))
		  
			col_name = model_group_names[group]
			test_dataframes[i][col_name] = test_dataframes[i]['device_model'].map(lambda x: ((model_counter[x]  + overall_prob*alpha) / (model_total_counter[x] + alpha) if x in models else 0.0))
	   

			# test_dataframes[i]['mean_age_brand'] = test_dataframes[i]['phone_brand'].map(lambda x: mean_age_brand[x] if x in brands else global_mean_age) / 100.0
			# test_dataframes[i]['mean_gender_brand'] = test_dataframes[i]['phone_brand'].map(lambda x: mean_gender_brand[x] if x in brands else global_mean_gender)
			# test_dataframes[i]['mean_age_model'] = test_dataframes[i]['device_model'].map(lambda x: mean_age_model[x] if x in models else global_mean_age) / 100.0
			# test_dataframes[i]['mean_gender_model'] = test_dataframes[i]['device_model'].map(lambda x: mean_gender_model[x] if x in models else global_mean_gender)
	
	print ('-'*100,'train',train)
	print ('-'*100,'test',test_dataframes[0])

	gc.collect()
	return train, test_dataframes, brand_group_names  + model_group_names #+ ['mean_age_brand', 'mean_gender_brand', 'mean_age_model', 'mean_gender_model']

def get_hash_data_memory_efficient(train, test, colnames_to_hash, type = 'tfidf'):
	if type == 'tfidf':
		tfv = TfidfVectorizer(min_df=1)
	elif type == 'tfidf_modified':
		tfv = TfidfVectorizer(min_df=1, ngram_range=(1,1),sublinear_tf = True,binary = True, norm = 'l2')
	else:
		tfv = CountVectorizer(min_df = 1, binary  = 1)

	c = 0
	for colname in colnames_to_hash:
		print (colname)
		colname_to_hash = [colname]
		#train_sub = train[colname_to_hash].astype(np.str).apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')
		#test_sub = test[colname_to_hash].astype(np.str).apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')
		
		print(train[colname])

		tfv.fit(train[colname].astype(np.str).fillna('Missing'))
		train_tf = tfv.transform(train[colname].astype(np.str).fillna('Missing'))
		test_tf = tfv.transform(test[colname].astype(np.str).fillna('Missing'))
		
		if c == 0:
			full_train_tf = train_tf
			full_test_tf = test_tf
		else:
			full_train_tf = sparse.hstack([full_train_tf, train_tf])
			full_test_tf = sparse.hstack([full_test_tf, test_tf])			

			print (colname, full_train_tf.shape, full_test_tf.shape)

		c += 1

	return full_train_tf, full_test_tf

def get_hash_data_simple(train, test, colnames_to_hash, type = 'tfidf'):
	if type == 'tfidf':
		tfv = TfidfVectorizer(min_df=1)
	elif type == 'tfidf_modified':
		tfv = TfidfVectorizer(min_df=1, ngram_range=(1,1),sublinear_tf = True,binary = True, norm = 'l2')
	elif type == 'counter':
		tfv = CountVectorizer(min_df = 1, binary  = 1)
	elif type == 'counter2':
		tfv = CountVectorizer(min_df = 1, binary  = False)

	train = train[colnames_to_hash].astype(np.str).apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')
	print ('-'*30,'df for hashing','\n',train)
	
	#df_tfv = tfv.fit_transform(df)
	tfv.fit(train)

	return tfv
	test = test[colnames_to_hash].astype(np.str).apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')
	test = tfv.transform(test)
	
	print ('-'*30,'df hashed','\n',train.shape)
		
	return train, test

def get_hash_data(train, test, colnames_to_hash, colnames_not_to_hash = [], type = 'tfidf'):

	df_all = pd.concat((train, test), axis=0, ignore_index=True)
	df = train

	split_len = len(train)

	if type == 'tfidf':
		tfv = TfidfVectorizer(min_df=1)
	elif type == 'tfidf_modified':
		tfv = TfidfVectorizer(min_df=1, ngram_range=(1,1),sublinear_tf = True,binary = True, norm = 'l2')
	else:
		tfv = CountVectorizer(min_df = 1, binary  = 1)

	df = df[colnames_to_hash].astype(np.str).apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')
	df_all = df_all[colnames_to_hash].astype(np.str).apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')
	print ('-'*30,'df for hashing','\n',df)
	#df_tfv = tfv.fit_transform(df)
	tfv.fit(df)
	del df
	gc.collect()
	df_tfv = tfv.transform(df_all)
	print(df_all)
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
	#print(sparse.find(train)[1][0:100],sparse.find(train)[2][0:100])

	return train, test


def find_params_xgb(train, Y, type = 'linear', folds = 5, total_evals = 50, sparse = True):

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
			
			#importance = model.get_fscore()
			#importance = sorted(importance.items(), key = lambda x: x[1], reverse = True)
			#print(importance[0:30])
			#if CVs == 0 and score >= 2.10:
			#	CVs = 1
			#	CV_score = 1000
			#	break
			CVs += 1
			CV_score += score

			print ('FOLD', CVs, 'SCORE', score)

		CV_score /= float(CVs)
		print ("\tCV_score {0}\n\n".format(CV_score))

		return {'loss': CV_score, 'status': STATUS_OK}

	def optimize(trials):
		if type == 'linear':
			space = {
					#'eta': hp.quniform('eta', 0.01, 0.03, 0.01),
					'eta':0.01,
					
					'lambda' : hp.quniform('lambda', 0.1, 0.7, 0.15),
					'lambda_bias' : hp.quniform('lambda_bias', 0.1, 0.7, 0.15),
					'alpha' : hp.quniform('alpha', 0.1, 0.5, 0.1),
					'booster': 'gblinear',
					'num_class' : 12,
					'eval_metric': 'mlogloss',
					'objective': 'multi:softprob',
					'nthread' : 8,
					'silent' : 0,
					'seed': 7
					 }

		if type == 'linear_reg':
			space = {
					#'eta': hp.quniform('eta', 0.01, 0.03, 0.01),
					'eta':0.01,
					
					'lambda' : hp.quniform('lambda', 0.1, 0.7, 0.15),
					'lambda_bias' : hp.quniform('lambda_bias', 0.1, 0.7, 0.15),
					'alpha' : hp.quniform('alpha', 0.1, 0.5, 0.1),
					'booster': 'gblinear',
					'eval_metric': 'rmse',
					'objective': 'reg:linear',
					'nthread' : 8,
					'silent' : 0,
					'seed': 7
					 }

		elif type=='tree':

			space = {
					'eta':0.003,
					#'eta' : hp.quniform('eta', 0.05, 0.2, 0.03),
					'colsample_bytree' : hp.quniform('colsample_bytree', 0.3, 0.9, 0.1),
					'subsample' : hp.quniform('subsample', 0.6, 0.9, 0.1),
					'max_depth' : hp.quniform('max_depth', 3, 9, 1),
					'booster': 'gbtree',
					'num_class' : 12,
					'eval_metric': 'mlogloss',
					'objective': 'multi:softprob',
					'nthread' : 8,
					'silent' : 1,
					'seed': 7
					 }

		elif type=='tree_reg':

			space = {
					'eta':0.1,
					#'eta' : hp.quniform('eta', 0.05, 0.2, 0.03),
					'colsample_bytree' : hp.quniform('colsample_bytree', 0.3, 0.9, 0.1),
					'subsample' : hp.quniform('subsample', 0.6, 0.9, 0.1),
					'max_depth' : hp.quniform('max_depth', 2, 8, 1),
					'booster': 'gbtree',
					'eval_metric': 'rmse',
					'objective': 'reg:linear',
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
	#print(Y)
	model = xgb.train(params, dtrain, 2000, evals=watchlist, verbose_eval=verbose_eval, early_stopping_rounds = 1)
	score = model.best_score

	return model, score

def predict_xgb(X, model, sparse = False):
	if sparse:
		preds = model.predict(xgb.DMatrix(X.tocsc(), missing = np.nan))
	else:
		preds = model.predict(xgb.DMatrix(X, missing = np.nan))
	

	return preds

def CV_evaluation(X,Y,model, folds = 5, sparse = False):
	
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
		preds = model.predict_proba(X_val)
		score = log_loss(y_val, preds)

		CVs += 1
		CV_score += score

		print ('FOLD', CVs, 'SCORE', score)

	CV_score /= float(CVs)
	print ("\tCV_score {0}\n\n".format(CV_score))
	return CV_score

def create_features_for_stack(X, Y, test_X, model, type='multiclass_classification', features_to_keep = 'all', nfolds = 5, sparse_matrix = True, seed = 1, n_classes=None):

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
			#print(cur_preds.shape)
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


	
	#print('X for stacking', new_X_train)

	return new_X_train, new_X_test, CV_score

def create_stack(X, Y, test_X, models, type='multiclass_classification', n_classes = None, features_to_keep = 0, nfolds = 5, sparse_matrix = True):
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

from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adadelta
from keras.layers.advanced_activations import LeakyReLU,PReLU,ELU

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
			#print (counter, number_of_batches)
			
		yield X_batch, y_batch
		if (counter == number_of_batches):
			if shuffle:
				np.random.shuffle(sample_index)
			counter = 0

def generate_model(params,type='multiclass_classification', n_classes = None):
		# create model
	hidden_sizes, input_dim, dropouts, init = params['hidden_sizes'], params['input_dim'], params['dropouts'], params['init']
	print (hidden_sizes, input_dim, dropouts, init)

	n_layers = len(hidden_sizes)

	model = Sequential()

	for l in range(n_layers):
		if l == 0:
			model.add(Dense(hidden_sizes[l], input_dim=input_dim, init=init))
		else:		
			model.add(Dense(hidden_sizes[l], init=init))
		
		#model.add(BatchNormalization())
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
			

	# Compile model
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

	np.random.seed(7)

	skf = StratifiedKFold(Y, n_folds=5, random_state = 1)
	batch_size = 256
	nb_epoch = 2
	
	CV_score = 0.0
	CVs = 1
	
	for train_index, val_index in skf:
		cur_train = train_X.tocsr()[train_index,:]
		cur_val = train_X.tocsr()[val_index,:]
		#Y_train=Y[train_index]
		#Y_val = Y[val_index]
		Y_train = np_utils.to_categorical(Y[train_index])
		Y_val = np_utils.to_categorical(Y[val_index])
		
		val_size = batch_size*(cur_val.shape[0]//batch_size)
		cur_val = cur_val[:val_size]
		Y_val = Y_val[:val_size]
		#for i in range(nb_epoch):
		#	cur_train = sparse.vstack([cur_train, cur_train])
		#	Y_train = np.vstack([Y_train, Y_train])
		batch_size = params['batch_size']
		nb_epoch = params['nb_epoch']

		model = generate_model(params,type=type)
		model.fit_generator(generator=batch_generator(cur_train, Y_train, batch_size, True),
			nb_epoch=nb_epoch, 
			samples_per_epoch=cur_train.shape[0])
		probas = model.predict_generator(generator=batch_generator(cur_val, Y_val, batch_size, False), val_samples = cur_val.shape[0])
		#print(probas)
		#probas = model.predict_on_batch(cur_val, batch_size=32)
		
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
	#print (model)

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


def select_features(train_X, y, test_X):
	pass

class GenderAgeGroupProb(object):
	def __init__(self, prior_weight=10.):
		self.prior_weight = prior_weight
	
	def fit(self, df, by):
		self.by = by
		#self.label = 'pF_' + by
		self.prior = df['group'].value_counts().sort_index()/df.shape[0]
		# fit gender probs by grouping column
		c = df.groupby([by, 'group']).size().unstack().fillna(0)
		total = c.sum(axis=1)
		self.prob = (c.add(self.prior_weight*self.prior)).div(c.sum(axis=1)+self.prior_weight, axis=0)
		return self
	
	def predict_proba(self, df):
		pred = df[[self.by]].merge(self.prob, how='left', 
								left_on=self.by, right_index=True).fillna(self.prior)[self.prob.columns]
		pred.loc[pred.iloc[:,0].isnull(),:] = self.prior
		return pred.values
	
def score(ptrain, by, prior_weight=10.):
	kf = KFold(ptrain.shape[0], n_folds=10, shuffle=True, random_state=0)
	pred = np.zeros((ptrain.shape[0],n_classes))
	for itrain, itest in kf:
		train = ptrain.iloc[itrain,:]
		test = ptrain.iloc[itest,:]
		ytrain, ytest = y[itrain], y[itest]
		clf = GenderAgeGroupProb(prior_weight=prior_weight).fit(train,by)
		pred[itest,:] = clf.predict_proba(test)
	return log_loss(y, pred)

