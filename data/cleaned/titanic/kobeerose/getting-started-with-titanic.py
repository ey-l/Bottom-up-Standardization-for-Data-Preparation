#!/usr/bin/env python
# coding: utf-8

# <h2 style="font-weight: bold">Titanic Competition</h2>
# 
# <h4>This is my first published notebook ever, So yeah it can't be about something other than the Titanic Competition 😄<br><br>I will be doing a simple then advanced EDA, Data Visualization and Pre-Processing. I also will test different models and techniques to improve my score.<br></h4>
# 
# * <h5 style="font-weight: 700">Your feedback is very welcome</h5>
# * <h5 style="font-weight: 700">If you find this notebook useful, please don't forget to upvote it!</h5>
# 

# In[ ]:


# Required packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


# In[ ]:


train = pd.read_csv('data/input/titanic/train.csv')
test = pd.read_csv('data/input/titanic/test.csv')


# In[ ]:


train.head()


# #  **Exploratory Data Analysis**

# In[ ]:


# Getting to know data
print(train.shape)
print(test.shape)


# In[ ]:


# summary of numerical variable
train.describe()


# In[ ]:


train.describe()


# In[ ]:


# summary of categorial variable
train.info()


# In[ ]:


test.info()


# In[ ]:


# checking null values
print("training data\n",train.isnull().sum())
print("\ntesting data\n",test.isnull().sum())


# In[ ]:


# let's clean visualizations :)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


# Visualization
count_plt = sns.countplot(train["Survived"])


Sex_plt = sns.countplot(x= "Survived",data=train, hue="Sex")


Embarked_plt = sns.countplot(x="Survived", data=train, hue="Embarked")


Pclass_plt = sns.countplot(x="Survived", data=train, hue="Pclass")


SibSp_plt = sns.boxplot(x="SibSp", y= "Survived", data=train)


Parch_plt = sns.boxplot(x="Parch", y= "Survived", data=train)


Age_plt = sns.distplot(train["Age"])



# In[ ]:


# Correlation heatmap
f, ax = plt.subplots(figsize=(10, 8))
corr = train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool_),linewidths=0.1,annot=True, cmap=sns.diverging_palette(150, 10, as_cmap=True),
            square=True, ax=ax)


# # **Pre-Processing**

# In[ ]:


# age mean, fair mean
train['Age'].fillna(train['Age'].mean(), inplace = True)
test['Age'].fillna(train['Age'].mean(), inplace = True)
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
train['Embarked'].fillna('S', inplace = True)


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


# dropping cabin column
train.drop(columns=["Cabin", "Name", "Ticket"], axis=1, inplace=True)
test.drop(columns=["Cabin", "Name", "Ticket"], axis=1, inplace=True)

# dropping passengerid
train = train.drop(['PassengerId'],axis=1)


# In[ ]:


# changing gender to numeric
train.loc[train.Sex=='female','Sex']=1
train.loc[train.Sex=='male','Sex']=0
train["Sex"] = train["Sex"].astype(str).astype(float)

# changing strings to numeric
train.loc[train.Embarked =='S','Embarked']= 3
train.loc[train.Embarked =='C','Embarked']=2
train.loc[train.Embarked =='Q','Embarked']=1
train["Embarked"] = train["Embarked"].astype(str).astype(float)


# In[ ]:


# Same for test data
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
test.loc[test.Sex=='female','Sex']=1
test.loc[test.Sex=='male','Sex']=0
test["Sex"] = test["Sex"].astype(str).astype(float)
test.loc[test.Embarked =='S','Embarked']= 3
test.loc[test.Embarked =='C','Embarked']=2
test.loc[test.Embarked =='Q','Embarked']=1
test["Embarked"] = test["Embarked"].astype(str).astype(float)
test.isnull().sum()


# In[ ]:


# checking data
print(train.head())
print(test.head())
print(train.corr())


# In[ ]:


train.info()


# In[ ]:


test.info()


# # **Training and Predicting**

# In[ ]:


# Classification Methods

# Logistic Regression
train_x= train.drop(columns=["Survived"], axis=1)
train_y= train["Survived"]

test_x= test.drop("PassengerId",axis=1)

logistic = LogisticRegression(solver='liblinear')