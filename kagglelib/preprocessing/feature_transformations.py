import numpy as np
from scipy import sparse
import pandas as pd

def normalize_data(train, test, max_norm = False, is_sparse = False):
    """create arithmetic combinations of order 2 or order 3 for numerical features

    Parameters
    ----------
    train : pandas.Dataframe or numpy.ndarray
        train dataset
    test : pandas.Dataframe or numpy.ndarray
        test dataset
    max_norm : boolean, optional
        whether to scale x to [-1, 1] values
        False by default
    is_sparse : boolean, optional
        whether input is scipy.sparse matrix 
        False by default

    Returns
    -------
    normalized train, test
    """
    
    if is_sparse:
        train = train.toarray()
        test = test.toarray()
    
    for i in xrange(train.shape[1]):
        mean_train = np.mean(train[:,i])
        std_train = np.std(train[:,i])
        
        train[:,i] -= mean_train
        test[:,i] -= mean_train

        train[:,i] /= std_train
        test[:,i] /= std_train

        if max_norm:
            max = np.max(abs(train[:,i]))
            train[:, i] /= max
            test[:, i] /= max
        
    if is_sparse:
        train = sparse.coo_matrix(train)
        test = sparse.coo_matrix(test)

    return train,test