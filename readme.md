My library of most useful functions to participate in Kaggle competitions. 

Supports types of models:
1. Scikit-Learm models
2. XGBoost models (defined as dicts of parameters)
3. Keras models (defined as dicts of parameters)

Supports tasks:
1. Binary classification
2. Multiclass classification
3. Regression

Contains functions for:
1. Data hashing (incl. memory-efficient versions)
2. Training XGBoost models (both trees and linear) and predicting
3. Training Keras models and predicting
4. Finding best params of XGBoost models by HyperOpt library
5. Cross-validation performance evaluation (Scikit-Learn models, Keras models, XGBoost models)
6. Building stacks of models
