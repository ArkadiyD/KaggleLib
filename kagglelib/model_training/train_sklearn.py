def train_sklearn(train_x, train_y, model):
    '''trains sklearn models
    
    Parameters
    ----------
    train_x : numpy.ndarray or scipy.sparse matrix
        train dataset
    train_y : numpy.ndarray
        target
    model: Model
        instance of Model class

    Returns 
    -------
    trained sklearn model
    '''

    model.model.set_params(**model.params)
    model.model = model.model.fit(train_x, train_y)
    return model.model