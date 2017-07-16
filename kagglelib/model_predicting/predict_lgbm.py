import pandas as pd

def predict_lgbm(val_x, model):
    ''' makes predictions by lightgbm model

    Parameters
    ----------
    val_x : numpy.ndarray or pandas.DataFrame
        dataset to make predictions
    model : lightgbm.Booster
        trained lightgbm trees ensemble

    Returns
    -------
    predictions array
    '''

    if isinstance(val_x, pd.DataFrame):
        x = val_x.values
    else:
        x = val_x

    preds = model.predict(x, num_iteration = -1)
    return preds