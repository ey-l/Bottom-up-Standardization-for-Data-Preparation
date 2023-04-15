#!/usr/bin/env python
# coding: utf-8

# This is a very basic code for absolute beginners. 
# It will be easy to understand and implement as i have not done anything huge but this code can give you a good introduction to what logistic regression is about and what are the things that had to be done before implementaion of the model.
# 
# ******So lets get started.

# In[ ]:


# importing important packages
import pandas as pd
import numpy as np


# In[ ]:


# reading the training data set
data = pd.read_csv('data/input/titanic/train.csv')


# In[ ]:


# take a good look at data
data


# In[ ]:


# by looking at data i conluded - some of the data columns are of no use for prediction
# that might not be true , maybe they can play a role in prediction but let's just assume
# they wont. 
# So we'll drop those columns
model_data = data.drop(['Name', 'Ticket', 'Fare', 'Cabin'], axis=1)


# In[ ]:


# dropping the rows with null values 
model_data.dropna(inplace=True)


# In[ ]:


# checking unique values for all categorical columns
print(model_data['Sex'].unique())
print(model_data['Pclass'].unique())
print(model_data['Embarked'].unique())
print(model_data['SibSp'].unique())
print(model_data['Parch'].unique())


# In[ ]:


model_data['Sex'].value_counts()


# In[ ]:


model_data['Pclass'].value_counts()


# In[ ]:


model_data['Embarked'].value_counts()


# In[ ]:


model_data['SibSp'].value_counts()


# In[ ]:


model_data['Parch'].value_counts()


# In[ ]:


# now we are doing one hot encoding of the data
# one hot encoding is a technique used to create multiple columns out of categorical columns
model_data = pd.get_dummies(model_data, columns = ['Sex'])
model_data = pd.get_dummies(model_data, columns = ['Pclass'])
model_data = pd.get_dummies(model_data, columns = ['Embarked'])
model_data = pd.get_dummies(model_data, columns = ['Parch'])
model_data = pd.get_dummies(model_data, columns = ['SibSp'])


# In[ ]:


# look at the data now, after one hot encoding.
model_data


# In[ ]:


# now we are declaring our dependent and independent variables.
y = model_data['Survived']
X = model_data.drop(['Survived', 'PassengerId'], axis=1)


# In[ ]:


# that is how we implement logistic regression
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='liblinear', random_state=0)