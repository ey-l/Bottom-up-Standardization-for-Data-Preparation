#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


data_train = pd.read_csv("data/input/titanic/train.csv", index_col ='PassengerId')


# In[ ]:


(data_train.isnull().sum()*100)/data_train.shape[0]


# In[ ]:


data_train = data_train.drop(["Name", "Ticket", "Cabin"], axis=1)


# In[ ]:


X = data_train.drop("Survived", axis=1)
y = data_train["Survived"]


# In[ ]:


numerical = [col for col in X.select_dtypes(exclude='object')]
categorial = [col for col in X.select_dtypes(include='object')]


# In[ ]:


numerical_pipeline = Pipeline(steps=[
                    ('impute', SimpleImputer(strategy='constant'))
])


# In[ ]:


categorial_pipeline = Pipeline(steps = [
                                ("impute_cat", SimpleImputer(strategy="most_frequent")),
                                ("encode",OneHotEncoder(handle_unknown='ignore') )
])


# In[ ]:


preprocessing = ColumnTransformer(transformers=[
                            ("num", numerical_pipeline, numerical),
                            ("cat", categorial_pipeline, categorial)
])


# In[ ]:


model_2 = GradientBoostingClassifier() 
pipeline_2 = Pipeline(steps = [
                            ("preprocessing", preprocessing),
                            ("model",model_2)
])


# In[ ]:


param = {
        'model__n_estimators': np.arange(50, 1000, 100)
}


# In[ ]:


grid = GridSearchCV(pipeline_2, param_grid = param, cv=5)


# In[ ]:

