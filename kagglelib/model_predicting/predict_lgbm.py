import pandas as pd

def predict_lgbm(val_x, model):
    """ makes predictions by lightgbm model

    Parameters
    ----------
    val_x : numpy.ndarray or pandas.DataFrame
        dataset to make predictions
    model : lightgbm.Booster
        trained lightgbm trees ensemble

    Returns
    -------
    predictions array
    """

    preds = model.predict(val_x, num_iteration = -1)
    return preds