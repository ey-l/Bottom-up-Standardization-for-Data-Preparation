#!/usr/bin/env python
# coding: utf-8

# # Titanic explainability: Why me? asks Miss Doyle
# 
# Sometimes pure predictive capacity is not the only thing we ask of our machine learning models, and increasingly it is just as important to be able explain how we arrived at our predictions.
# 
# ### Explainability and the GDPR
# Being able to easily explain how a model works, or how a decision was made based on the model, is not a mere intellectual nicety; in fact the [EU General Data Protection Regulation (GDPR) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679), Article 15(1)(h) states:
# 
# > "*The data subject shall have the right to obtain... ...meaningful information about the logic involved, as well as the significance and the envisaged consequences of such processing*"
# 
# also, in Article 22:
# 
# > "*The data subject shall have the right to obtain... ...human intervention on the part of the controller, to express his or her point of view and to contest the decision.*"
# 
# In order to comply with this, the data scientist must be able to clearly explain how any decision was originally arrived at. 
# 
# Non-compliance with the GDPR by a company can result in serious consequences, and it is part of the job of a data scientist to mitigate such risks for their employers. (For those interested the website [GDPR Enforcement Tracker](https://www.enforcementtracker.com/) has a partial list of fines that have been imposed).
# 
# The following is a small demonstration script which applies the [logistic regression classifier from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) to the [Titanic data set](https://www.kaggle.com/c/titanic). 

# In[ ]:


#!/usr/bin/python3
# coding=utf-8
#===========================================================================
# This is a minimal script to perform a classification 
# using the logistic regression classifier from scikit-learn 
# Carl McBride Ellis (18.IV.2020)
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
# features we use
#===========================================================================
features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]

#===========================================================================
# for the features that are categorical we use pd.get_dummies:
# "Convert categorical variable into dummy/indicator variables."
#===========================================================================
X_train  = pd.get_dummies(train_data[features])
y_train  = train_data["Survived"]
X_test   = pd.get_dummies(test_data[features])

#===========================================================================
# perform the classification
#===========================================================================
from sklearn.linear_model import LogisticRegression
# we use the default Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm
classifier = LogisticRegression(solver='lbfgs',fit_intercept=False)