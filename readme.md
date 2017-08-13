# KaggleLib
Library contains most useful functions to participate in Kaggle and other machine learning competitions. 
Module is divided into parts usually included in machine learning application pipeline: 
**Preprocessing -> Features engineering -> Features selection -> Model tuning -> Model training -> Model predicting -> Building ensemble of models**

# Requirements
All dependencies listed in ```requirements.txt```

# Installation and setup
0. clone repository
1. ```cd KaggleLib```
2. ```pip install -r requirements.txt```
3 ```pip install .```
4. check installation by running example: ```python examples/example.py```

# API
The library contains following parts:

0. **Model** - generic class for machine learning models
    * ```type``` : model type (XGBoost, LightGBM, Keras or Scikit-Learn) 
    * ```params``` : dictionary of model parameters
    * ```model``` : instance of model object
    * ```cv_score``` : cross-validation score
    
1. **Preprocessing**
    * ```hash_data``` : hashing of categorical columns (one-hot)
    * ```normalize_data``` : numerical data normalization
    
2. **Feature engineering**
    * ```make_numerical_interactions``` : feature interactions of 2 and 3 order, operations: sum, division, multipliciation, division
    * ```make_categorical_interactions``` : categorical feature interactions of 2 and 3 order
    * ```categorical_target_encoding```: 
    * ```logarithm``` : log feature transformation
    * ```exponent``` : exponent feature transformation
    * ```sigmoid``` : sigmoid feature transformation
    * ```trgonometry``` : sin, cos, tan feature transformation
    
3. **Feature selection**
    * ```genetic_feature_selection``` : select subset of features with best cross-validation metric by genetic algorithm (evolutional change of features subsets)

4. **Model tuning**
    * ```cross_validation``` : calculate cross-validation score of a model
    * ```tune_lgbm``` : find best LightGBM parameters by HyperOpt
    * ```tune_xgb``` : find best XGBoost parameters by HyperOpt

5. **Model training**
    * ```train_keras``` : train Keras model
    * ```train_lgbm``` : train LightGBM model
    * ```train_xgb``` : train XGBoost model
    * ```train_sklearn``` : train Scikit-Learn mdoel

6. **Model predicting**
    * ```predict_keras``` : prediction by Keras model
    * ```predict_lgbm``` : prediction by LightGBM model
    * ```predict_xgb``` : prediction by XGBoost model
    * ```predict_sklearn``` : prediction by Scikit-Learn model

7. **Model ensembles**
    * ```stacking``` : creating stack of model using out-of-fold predictions technique

8. **Utils**
    * ```make_folds``` : split data into folds
    * ```generate_keras_model``` : generate Keras model by dictionary
    * ```HistoryCallback``` : callback to preserve Keras training information on every epoch