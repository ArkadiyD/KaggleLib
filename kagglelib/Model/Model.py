class Model:
    '''generic wrapper for Model instance 
    
    allows to select epoches based on validation metrics on them and aggregate predictions for test on selected epoches

    Parameters
    ----------
    type : string, 'sklearn' or 'xgboost' or 'lightgbm' or 'keras'
        model type
    params : dict, optional
        dictionary of model parameters
        None by default
    
    Attributes
    -------
    type : string
        model type
    params : dict
        dictionary of model parameters
    model : xgboost model or lightgbn model or keras model or skelarn model
    	fitted instance of model of corresponding type
    cv_score : float
        score on cross-validation     
    '''
    
    def __init__(self, type, params = None):
        self.type = type
        self.params = params
        self.model = None
        self.cv_score = None
        if self.type != 'sklearn' and self.type != 'xgboost' and self.type != 'lightgbm' and self.type != 'keras':
            raise ValueError("model type should be 'xgboost' or 'lightgbm' or 'sklearn' or 'keras', got %s" % type)

    def __str__(self):
        return 'type: ' + str(self.type) + '\nparams: ' + str(self.params) + '\ncv_score: ' + str(self.cv_score) + '\nmodel: ' + str(self.model)