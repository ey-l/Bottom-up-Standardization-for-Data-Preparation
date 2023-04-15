#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


# importing datasets
train_dataset = pd.read_csv("data/input/titanic/train.csv")
test_dataset = pd.read_csv("data/input/titanic/test.csv")
y_test_dataset = pd.read_csv("data/input/titanic/gender_submission.csv")


# In[ ]:


# creating train/test and x/y splits
x_train = train_dataset.iloc[:, [2, 4, 5, 6, 7]].values
y_train = train_dataset.iloc[:, 1].values
x_test = test_dataset.iloc[:, [1, 3, 4, 5, 6]].values
y_test = y_test_dataset.iloc[:, 1].values


# In[ ]:


# labeling genders
le1 = LabelEncoder()
x_train[:, 1] = le1.fit_transform(x_train[:, 1])
x_test[:, 1] = le1.fit_transform(x_test[:, 1])


# In[ ]:


# dealing with nan data
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
x_train_imp = imp.fit_transform(x_train)
x_test_imp = imp.fit_transform(x_test)


# In[ ]:


# feature scaling
sc = StandardScaler()
x_train_imp = sc.fit_transform(x_train_imp)
x_test_imp = sc.transform(x_test_imp)


# In[ ]:


# ----------------------------- RANDOM FOREST CLASSIFIER ---------------
# training using entropy criterion
rfc = RandomForestClassifier(n_estimators=10000, criterion='entropy', n_jobs=-1)