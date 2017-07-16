def predict_keras(val_x, model, eval_metric):
    ''' makes predictions by keras neural net model

    Parameters
    ----------
    val_x : numpy.ndarray
        dataset to make predictions
    model : keras.Model
        trained neural net
    eval_metric : string, possible variants: 'mse', rmse', 'auc', 'logloss', 'mlogloss', 'error', 'merror' 
        eval_metric for model

    Returns
    -------
    predictions array
    '''

    preds_val = model.predict(val_x)
        
    if eval_metric == 'auc' or eval_metric == 'rmse' or eval_metric == 'mse' or eval_metric == 'error':
        preds_val = preds_val[:, 0]

    return preds_val