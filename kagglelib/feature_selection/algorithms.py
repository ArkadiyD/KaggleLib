from ..model_tuning import cross_validation
import numpy as np
import pandas as pd

def genetic_feature_selection(train_x, train_y, task, model, folds, eval_metric, categorical_features = [], n_classes = 2, sparse = False, iterations = 100, generation_size = 20, generation_best_ratio = 10, mutation_prob = 0.05, stopping_rounds = -1, epochs = None, average_epochs = None, missing = np.nan, verbose = False):
    """genetic algorithm for features selection
    
    all features are encoded  as binary vector

    Parameters
    ----------
    train_x : pandas.DataDrame or numpy.ndarray
        train dataset
    train_y : np.ndarray
        target
    task : string, 'regression' or 'binary_classification' or 'multiclass_classification'
        task to solve
    model : Model
        Model  instance
    folds : list of pairs of lists
        indices of train and validation folds
        folds[i][0] - train indices in i-th train-val split
        folds[i][1] - validation indices in i-th train-val split
    eval_metric : string, possible variants: 'mse', rmse', 'auc', 'logloss', 'mlogloss', 'error', 'merror' 
        eval_metric for model
    categorical_features : list of strings or lists of integers, optional
        column names (if train is Pandas DataFrame) or column indices (if train is Numpy array) of categorical features
        [] by default
    n_classes : integer, optional
        number of classes in case of classification task
        2 by default
    sparse : boolean, optional
        whether train is sparse Scipy matrix
        False by default
    iterations : integer, optional
        number of iterations in genetic algorithm
        100 by default
    generation_size : integer, optional
        number of objects in one generation
        20 by default
    generation_best_ratio : integer, optional
        number of best objects to survive to next generation
        10 by default
    generation_best_ratio : float, optional
        mutation probability
        0.05 by default
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
    verbose : , optional
        whether to print running info
        False by default

    Returns 
    -------
    list of best features, best CV score  (CV score with selected features)
    """
    
    np.random.seed(1)

    n_features = train_x.shape[1]

    if isinstance(train_x, pd.DataFrame):
        train_values = train_x.values
        features_list = train_x.columns.values
    else:
        features_list = np.arange(n_features)
        train_values = train_x

    def score_features(features_sample):
            
        features_ind = np.where(features_sample == 1)[0]

        if model.type == 'xgboost':
            return cross_validation.CV_score_xgb(train_values[:, features_ind], train_y, model.params, eval_metric, folds, sparse = sparse, stopping_rounds = stopping_rounds, missing = missing, verbose = False)
        
        elif model.type == 'lightgbm':
            return cross_validation.CV_score_lgbm(train_values[:, features_ind], train_y, model.params, categorical_features, eval_metric, folds, stopping_rounds = stopping_rounds, verbose = False)

        elif model.type == 'keras':
            return cross_validation.CV_score_keras(train_values[:, features_ind], train_y, task, model.params, eval_metric, folds, n_classes = n_classes, stopping_rounds = stopping_rounds, epochs = epochs, average_epochs = average_epochs, verbose = False)

        elif model.type == 'sklearn':
            return cross_validation.CV_score_sklearn(train_values[:, features_ind], train_y, model, eval_metric, folds, verbose = False)

    generation = np.zeros((generation_size, n_features), dtype = np.int16)
    generation_scores = np.zeros((generation_size, ), dtype = np.float32)

    def random_sample():
        sample = np.zeros((n_features, ), dtype = np.int16)
        for i in xrange(n_features):
            if np.random.uniform(0, 1) < 0.5:
                sample[i] = 1
        return sample

    def crossover(parent_a, parent_b):
        child = np.copy(parent_a)
        for i in xrange(n_features):
            if np.random.uniform(0, 1) < 0.5:
                child[i] = parent_b[i]
        return child

    def mutation(sample):
        mutated_sample = np.copy(sample)
        for i in xrange(n_features):
            if np.random.uniform(0, 1) < mutation_prob:
                mutated_sample[i] ^= 1
        return mutated_sample

    def sort_generation(generation, generation_scores):
        
        sort_ind = np.argsort(generation_scores)
        if eval_metric == 'auc':
            sort_ind = sort_ind[::-1]

        generation = generation[sort_ind]
        generation_scores = generation_scores[sort_ind]

        return generation, generation_scores

    for i in xrange(generation_size):
        generation[i] = random_sample()
        generation_scores[i] = score_features(generation[i])

    for iter in xrange(iterations):

        generation, generation_scores = sort_generation(generation, generation_scores)
        if verbose:
            print iter, generation_scores
            print 'cur best: ', features_list[np.where(generation[0] == 1)[0]]

        for i in xrange(generation_best_ratio, generation_size, 1):
            parent_a = generation[np.random.randint(0, generation_best_ratio)]
            parent_b = generation[np.random.randint(0, generation_best_ratio)]
            child = crossover(parent_a, parent_b)
            mutated_child = mutation(child)
            generation[i] = np.copy(mutated_child)
            generation_scores[i] = score_features(generation[i])
            if verbose:
                print i, generation_scores[i]

    sort_generation(generation, generation_scores)
    best_features, best_score = generation[0], generation_scores[0]     
    best_features = features_list[np.where(best_features == 1)[0]]
    
    if verbose:
        print 'best features set: ', best_features
        print 'best score: ', best_score

    return best_features, best_score