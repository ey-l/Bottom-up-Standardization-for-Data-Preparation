#!/usr/bin/env python
# coding: utf-8

# ### Hyperparameter grid search sample code
# This is a sample code for performing a hyperparameter grid search using the [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) routine from scikit-learn. We shall use the default 5-fold [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics&#41;). Finally, for the classifier we shall use the [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), also from scikit-learn.
# 

# In[ ]:


#!/usr/bin/python3
# coding=utf-8
#===========================================================================
# This is a simple script to perform a classification on the kaggle 
# 'Titanic' data set using a grid search, in conjunction with a 
# random forest classifier
# Carl McBride Ellis (1.V.2020)
#===========================================================================
#===========================================================================
# load up the libraries
#===========================================================================
import pandas as pd
import numpy  as np

#===========================================================================
# read in the data
#===========================================================================
train_data = pd.read_csv('data/input/titanic/train.csv')
test_data  = pd.read_csv('data/input/titanic/test.csv')

#===========================================================================
# select some features of interest ("ay, there's the rub", Shakespeare)
#===========================================================================
features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]

#===========================================================================
# for the features that are categorical we use pd.get_dummies:
# "Convert categorical variable into dummy/indicator variables."
#===========================================================================
X_train       = pd.get_dummies(train_data[features])
y_train       = train_data["Survived"]
final_X_test  = pd.get_dummies(test_data[features])

#===========================================================================
# hyperparameter grid search using scikit-learn GridSearchCV
# we use the default 5-fold cross validation
#===========================================================================
from sklearn.model_selection import GridSearchCV
# we use the random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='gini', max_features='auto')
gs = GridSearchCV(cv=5, error_score=np.nan, estimator=classifier,
# dictionaries containing values to try for the parameters
param_grid={'min_samples_leaf':  [10, 15, 20],
            'max_depth':         [3, 4, 5, 6],
            'n_estimators':      [10, 20, 30]})