#!/usr/bin/env python
# coding: utf-8

# #Importing the Dependencies

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 


# #Data Collection & Processing

# Load the train data from csv 

# In[ ]:


titanic_data = pd.read_csv('data/input/titanic/train.csv')


# Printing the first 5 rows of the dataframe

# In[ ]:


titanic_data.head()


# Number of rows and Columns

# In[ ]:


titanic_data.shape


# Getting some informations about the data

# In[ ]:


titanic_data.info()


# Checking the unique values

# In[ ]:


titanic_data.nunique().sum


# Check the number of missing values in each column

# In[ ]:


titanic_data.isnull().sum()


# #Handling the Missing values

# Drop the "Cabin" column from the dataframe

# In[ ]:


titanic_data = titanic_data.drop(columns='Cabin', axis=1)


# Replacing the missing values in "Age" column with mean value

# In[ ]:


titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# Replacing the missing values in "Embarked" column with mode value

# In[ ]:


titanic_data.Embarked.mode()


# In[ ]:


titanic_data["Embarked"] = titanic_data["Embarked"].fillna("S")


# Check the number of missing values in each column

# In[ ]:


titanic_data.isnull().sum()


# #Data Analysis

# Getting some statistical measures about the data

# In[ ]:


titanic_data.describe()


# Finding the number of people survived and not survived

# In[ ]:


titanic_data['Survived'].value_counts()


# #Data Visualization

# Making a count plot for "Survived" column

# In[ ]:


sns.countplot('Survived', data=titanic_data)


# Making a count plot for "Sex" column

# In[ ]:


titanic_data['Sex'].value_counts()


# In[ ]:


sns.countplot('Sex', data=titanic_data)


# Number of survivors Gender wise

# In[ ]:


sns.countplot('Sex', hue='Survived', data=titanic_data)


# Making a count plot for "Pclass" column

# In[ ]:


sns.countplot('Pclass', data=titanic_data)


# In[ ]:


sns.countplot('Pclass', hue='Survived', data=titanic_data)


# #Encoding the Categorical Columns

# In[ ]:


titanic_data['Sex'].value_counts()


# In[ ]:


titanic_data['Embarked'].value_counts()


# # Converting Categorical Columns

# In[ ]:


titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[ ]:


titanic_data.head()


# #Separating features & Target

# In[ ]:


X = titanic_data.drop(columns = ['Name','Ticket','Survived'],axis=1)
y = titanic_data['Survived']


# #Splitting the data into training data & Test data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# #Model Training

# Training the Logistic Regression model with training data

# In[ ]:


lr_model = LogisticRegression()
rfc_model=RandomForestClassifier()
knn_model=KNeighborsClassifier(n_neighbors=15)


# Logistic Regression

# In[ ]:

