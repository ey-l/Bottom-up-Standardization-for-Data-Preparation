#!/usr/bin/env python
# coding: utf-8

# # **Inporting important Libraries**

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# # Preparing the Training data

# In[ ]:


train_data = pd.read_csv("data/input/titanic/train.csv")


# In[ ]:


train_data.head(5)


# # Analysis of Sex :
# **74.2% of the women survived**
# 
# **Only 18.8% of the men survived**

# In[ ]:


sns.countplot(x=train_data['Sex'], hue=train_data['Survived'])



# # **Analysis on the basis of Pclass**

# In[ ]:


sns.countplot(x=train_data['Pclass'], hue=train_data['Survived'])



# # Dropping useless columns & replacing null values in Embarked

# In[ ]:


train_data1 = train_data.drop(["PassengerId","Cabin","Name","Ticket","Parch"],axis=1)
train_data1['Embarked'] = train_data1['Embarked'].fillna('C')


# ****Converting string values in Sex and Embarked to 0s, 1s and 2s****

# In[ ]:


train_data1['Sex'].replace('female', 0,inplace=True)
train_data1['Sex'].replace('male', 1,inplace=True)
train_data1['Embarked'].replace('S', 0,inplace=True)
train_data1['Embarked'].replace('C', 1,inplace=True)
train_data1['Embarked'].replace('Q', 2,inplace=True)
train_data1["Age"].fillna("39", inplace = True)


# In[ ]:


train_data1.head(5)


# ****Dropping null values****

# In[ ]:


train_data1.dropna(inplace=True)


# In[ ]:


train_data1.isnull().sum()


# In[ ]:


train_data1.info()


# In[ ]:


x_train = train_data1.drop("Survived",axis=1)
y_train = train_data1["Survived"]


# # **Preparing the testing data:**

# In[ ]:


test_data = pd.read_csv("data/input/titanic/test.csv")


# In[ ]:


test_data.info()


# ****Dropping a few columns****

# In[ ]:


test_data1 = test_data.drop(["PassengerId","Cabin","Name","Ticket","Parch"],axis=1)


# In[ ]:


test_data1.info()


# ****Replacing null values in Age with mean age of the entire dataset****

# In[ ]:


test_data1["Age"].fillna("34", inplace = True)


# In[ ]:


test_data1.info()


# ****Replacing null values in fare with the mode fare of the dataset****

# In[ ]:


test_data1["Fare"].fillna("7.75", inplace = True)


# In[ ]:


test_data1.info()


# ****Converting String values in Sex and Embarked to integer representatives****

# In[ ]:


test_data1['Sex'].replace('female', 0,inplace=True)
test_data1['Sex'].replace('male', 1,inplace=True)
test_data1['Embarked'].replace('S', 0,inplace=True)
test_data1['Embarked'].replace('C', 1,inplace=True)
test_data1['Embarked'].replace('Q', 2,inplace=True)


# In[ ]:


x_test = test_data1


# # **Building the model**

# In[ ]:


log = LogisticRegression()


# ****Model fitting****

# In[ ]:

