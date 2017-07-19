import xgboost as xgb
import pandas as pd
import numpy as np

def predict_xgb(val_x, model, missing = np.nan):
    """ makes predictions by xgboost model

    Parameters
    ----------
    val_x : numpy.ndarray or scipy.sparse
        dataset to make predictions
    model : xgboost model
        trained xgboost
    missing : float or np.nan, optional
        which values are treated as missing by xgboost
        np.nan by default

    Returns
    -------
    predictions array
    """

    if isinstance(val_x, pd.DataFrame):
        x = val_x.values
    else:
        x = val_x
        
    dvalid = xgb.DMatrix(x, missing = missing)
    preds = model.predict(dvalid)
    return preds