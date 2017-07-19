def predict_sklearn(val_x, model, eval_metric, task):
    """ makes predictions by scikit-learn model

    Parameters
    ----------
    val_x : numpy.ndarray or scipy.sparse or pandas.DataFrame
        dataset to make predictions
    model : sklearn.model
        trained model
    task : string, 'regression' or 'binary_classification' or 'multiclass_classification'
        task to solve
    eval_metric : string, possible variants: 'mse', rmse', 'auc', 'logloss', 'mlogloss', 'error', 'merror' 
        eval_metric for model

    Returns
    -------
    predictions array
    """

    if eval_metric == 'logloss' or eval_metric == 'mlogloss' or eval_metric == 'auc':
        preds_val = model.predict_proba(val_x)

    else:
        preds_val = model.predict(val_x)

    if task == 'binary_classification':
        preds_val = preds_val[:, 1]

    return preds_val