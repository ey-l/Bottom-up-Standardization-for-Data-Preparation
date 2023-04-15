#!/usr/bin/env python
# coding: utf-8

# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTr0_0EtE1pjZl9xezJREPfPqCR0xKgHxvUGPDWPnMNh8wne_Rc)
# 
# # This notebook only aims to introduce the "pipeline" concept. 
# 
# ## So what is "pipeline" ? 
# 
# - Well, the pipeline is an abstract concept aiming to perform several different transformations before the final estimatior. The below is taken from [SKLEARN PIPELINE DOCUMENTATION](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
# 
# - The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters. For this, it enables setting parameters of the various steps using their names and the parameter name separated by a ‘__’, as in the example below. A step’s estimator may be replaced entirely by setting the parameter with its name to another estimator, or a transformer removed by setting it to ‘passthrough’ or None.
# 
# 
# **Parameters**:
# 
# - **steps**: (list) List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, with the last object an estimator.
# 
# - **memory**: (None, str or object with the joblib.Memory interface, optional) Used to cache the fitted transformers of the pipeline. By default, no caching is performed. If a string is given, it is the path to the caching directory. Enabling caching triggers a clone of the transformers before fitting. Therefore, the transformer instance given to the pipeline cannot be inspected directly. Use the attribute named_steps or steps to inspect estimators within the pipeline. Caching the transformers is advantageous when fitting is time consuming.
# 
# - **verbose**: (bool, default=False) If True, the time elapsed while fitting each step will be printed as it is completed.
# 
# **Attributes**:
# 
# - **named_steps**: (bunch object, a dictionary with attribute access) Read-only attribute to access any step parameter by user given name. Keys are step names and values are steps parameters.
# 
# 
# 
# 
# 
# 

# ## For this notebook, we will be using the Titanic Data
# 
# ![](https://res.cloudinary.com/dk-find-out/image/upload/q_80,w_1920,f_auto/MA_00079563_yvu84f.jpg)
# 
# 
# 

# # importing the libraries

# In[ ]:


import numpy as np 
import pandas as pd

from sklearn.pipeline      import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute        import SimpleImputer
from sklearn.linear_model  import LogisticRegression


# # loading the files

# In[ ]:


df_train = pd.read_csv('data/input/titanic/train.csv')
df_test  = pd.read_csv('data/input/titanic/test.csv')
df_sample= pd.read_csv('data/input/titanic/gender_submission.csv')


# # information about the dataset

# In[ ]:


df_train.info()


# In[ ]:


df_train.head()


# # Dropping unnecessary columns

# In[ ]:


df_train.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
df_test.drop( ['Name','Ticket','Cabin'],axis=1,inplace=True)

# We could create new features from these 3 but I aim to keep it simple and minimal


# # We both have numerical and categorical features

# In[ ]:


sex    = pd.get_dummies(df_train['Sex'],drop_first=True)
embark = pd.get_dummies(df_train['Embarked'],drop_first=True)

df_train = pd.concat([df_train,sex,embark],axis=1)
df_test  = pd.concat([df_test ,sex,embark],axis=1)

df_train.drop(['Sex','Embarked'],axis=1,inplace=True)
df_test.drop(['Sex','Embarked'],axis=1,inplace=True)


# # Only numerical features left to feed our pipeline
# 
# - Imputation
# - Standard Scaling
# - Predictive Model

# In[ ]:


imputer  = SimpleImputer()
scaler   = StandardScaler()
clf      = LogisticRegression()
pipe     = make_pipeline(imputer,scaler,clf)


# In[ ]:


features = df_train.drop('Survived',axis=1).columns

X,y   = df_train[features], df_train['Survived']
df_test.fillna(df_test.mean(),inplace=True)


# # Fitting the pipeline

# In[ ]:

