#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd


# In[ ]:


test = pd.read_csv('data/input/titanic/test.csv')
train = pd.read_csv('data/input/titanic/train.csv')


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


# Find number of nulls in each column
null_columns=train.columns[train.isnull().any()]
train[null_columns].isnull().sum()


# ### Define Predictors & Features
# 
# After watchin the movie, most of the ones who didn't make it off the boat were in the lower decks (third/second class). Suggesting the more money spent (Fare) gets a higher ticket class (Pclass).

# In[ ]:


# Prediction target
y = train.Survived
# Features (what's used to predict)
features = ['Fare', 'Pclass']
X = train[features]


# ## Make Model
# Logistic Regression is used because it better predicts binary (yes/no) fields.
# 
# 
# ### Regression Analysis
# - Logistic regression is a type of regression analysis. Regression analysis is a predictive modeling technique whicg finds the relationship between a dependent and independent variable(s). 
# - It is useful for predicting specific metrics like the number of units a company need to produce to meet demand. Another might be measuring the impact an independent variable has on the dependent.
# 
# Regresson analysis has two camps: **Linear regression** and **Logistic regression**
# 
# ### Linear Regression
# Is used for predictive analysis to find the extent of the relationship between the independent and dependent variables. It would be used to describe the effect ads would have on a company's revenue. It would output a trend line with ads on the x and revenue on the y.
# 
# 
# ### Logistic Regression
# **This a classification algorithm used to predict a binary outcome based on a set of independent variables**
# 
# In other words, only use this if the prediction is a yes/no, pass/fail, etc.
# 
# For the independent variables, the same does not apply. They can be:
# 1. Continuous - temperature, mass, price.
# 2. Discrete, ordinal - rated customer satisfaction (scaled from 1-5)
# 3. Discrete, nominal - fits into groups but there is no hierarchy (colours)

# In[ ]:


from sklearn.linear_model import LogisticRegression

# random_state controls the random number generator being used 
titanic_mdl = LogisticRegression(random_state=0)
