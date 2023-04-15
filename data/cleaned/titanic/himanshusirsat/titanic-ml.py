#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import xgboost


# # Readng Data

# In[ ]:


df =pd.read_csv("data/input/titanic/train.csv")
df


# # Exploratory Data analysis

# In[ ]:


sns.countplot(x="Sex",data=df)


# In[ ]:


sns.countplot(x="Sex",hue="Survived",data=df)


# In[ ]:


sns.distplot(x=df.Age,hist=True)


# In[ ]:


sns.distplot(x=df.Fare,hist=True)


# In[ ]:


sns.countplot(x="Pclass",hue="Survived",data=df)


# In[ ]:


sns.countplot(x="Survived",hue="Pclass",data=df)


# In[ ]:


sns.countplot(x="Embarked",data=df)


# In[ ]:


scatter_matrix(df)



# In[ ]:


print(df.corr())
sns.heatmap(df.corr())


# # Data Processing

# In[ ]:


df.isnull()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df["Age"].fillna(df["Age"].mean(),inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)


# In[ ]:


df.info()


# In[ ]:


data=df.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)
data


# In[ ]:


#from sklearn.preprocessing import OneHotEncoder
#ohe=OneHotEncoder()


# In[ ]:


dum=pd.get_dummies(data,columns=["Pclass","Sex","Embarked"])


# In[ ]:


#removing highly correlated variables 


# In[ ]:


print(dum.corr())


# In[ ]:


upd_dum=dum.drop(["Pclass_2","Sex_male","Embarked_Q"],axis=1)
#upd_dum["Fare"]=upd_dum["Fare"]/32
#upd_dum["Age"]=upd_dum["Age"]/30


# In[ ]:


upd_dum.info()


# In[ ]:


sns.heatmap(upd_dum.isnull(),cbar=False)


# In[ ]:


upd_dum


# In[ ]:


#upd_dum["scale_fare"]=upd_dum["Fare"]/20


# In[ ]:


#dummy=upd_dum.drop(["Fare"],axis=1)


# In[ ]:


X=upd_dum.drop(["Survived"],axis=1)
y=upd_dum["Survived"]


# # Model Selection

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


cross_val_score(RandomForestClassifier(n_estimators=65),X,y,cv=3).mean()


# In[ ]:


cross_val_score(LogisticRegression(),X,y,cv=3).mean()


# In[ ]:


cross_val_score(GaussianNB(),X,y,cv=3).mean()


# In[ ]:


cross_val_score(MultinomialNB(),X,y,cv=3).mean()


# In[ ]:


#This is the bst model
cross_val_score(xgboost.XGBClassifier(n_estimators=20),X,y,cv=3).mean()


# In[ ]:


model=xgboost.XGBClassifier(n_estimators=20)