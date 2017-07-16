My library of most useful functions to participate in Kaggle competitions. 

Supports types of models: 
*	Scikit-Learn models
* XGBoost models (defined as dicts of parameters)
* Keras models (defined as dicts of parameters)

Supports tasks:
*	Binary classification
2.	Multiclass classification
3.	Regression

Contains functions for:
* Data hashing (incl. memory-efficient versions)
2. Training XGBoost models (both trees and linear) and predicting
3. Building Keras models from dict of parameters, training them and predicting
4. Finding best params of XGBoost models by HyperOpt library
5. Cross-validation performance evaluation (Scikit-Learn models, Keras models, XGBoost models)
6. Building stacks of models
