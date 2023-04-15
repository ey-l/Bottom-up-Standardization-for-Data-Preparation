#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df_train = pd.read_csv('data/input/titanic/train.csv')


# In[ ]:


df_test = pd.read_csv('data/input/titanic/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_train['Cabin'].isnull().sum()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_train['Survived'].unique()


# In[ ]:


df_train['Age'].unique()


# In[ ]:


df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())


# In[ ]:


df_train['Embarked'].unique()


# In[ ]:


df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].median)


# In[ ]:


#df_train['Cabin'] = df_train['Cabin'].fillna(df_train['Cabin'].median)


# In[ ]:


df_train.head()


# In[ ]:


df_train=df_train.drop(labels='Cabin',axis=1)


# In[ ]:


df_test


# In[ ]:


df_test.isnull().sum()


# In[ ]:


df_test = df_test.drop(labels='Cabin',axis=1)


# In[ ]:


df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())


# In[ ]:


df_test.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()


# In[ ]:


df_train['Name'] = label.fit_transform(df_train['Name'])


# In[ ]:


df_train['Name']


# In[ ]:


df_train['Age'] = label.fit_transform(df_train['Age'])


# In[ ]:


df_train['Sex'] = label.fit_transform(df_train['Sex'])
df_train['SibSp'] = label.fit_transform(df_train['SibSp'])
df_train['Parch'] = label.fit_transform(df_train['Parch'])
df_train['Ticket'] = label.fit_transform(df_train['Ticket'])
df_train['Fare'] = label.fit_transform(df_train['Fare'])
#df_train['Cabin'] = label.fit_transform(df_train['Cabin'])


# In[ ]:


df_train['Embarked'] = label.fit_transform(df_train['Embarked'].astype(str))


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_test['Name'] = label.fit_transform(df_test['Name'])
df_test['Age'] = label.fit_transform(df_test['Age'])
df_test['Sex'] = label.fit_transform(df_test['Sex'])
df_test['SibSp'] = label.fit_transform(df_test['SibSp'])
df_test['Parch'] = label.fit_transform(df_test['Parch'])
df_test['Ticket'] = label.fit_transform(df_test['Ticket'])
df_test['Fare'] = label.fit_transform(df_test['Fare'])
df_test['Embarked'] = label.fit_transform(df_test['Embarked'].astype(str))


# In[ ]:


df_test.head()


# In[ ]:


x = df_train
target= df_test


# In[ ]:


x.head()


# In[ ]:


target.head()


# In[ ]:


X = x.drop(labels=["PassengerId",'Survived'],axis=1)
y = x['Survived']


# In[ ]:


X_scaled = scaler.fit_transform(X)
X_scaled


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.7,random_state=120)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
xg=xgb.XGBClassifier(random_state=1,learning_rate=0.01)