#!/usr/bin/env python
# coding: utf-8

# > # **MACHINE LEARNING FROM DISASTER**

# **Let's make survival prediction from the titanic dataset.**

# In[ ]:


#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Reading datasets
#train data and test data
train=pd.read_csv('data/input/titanic/train.csv')
test=pd.read_csv('data/input/titanic/test.csv')


# In[ ]:


#Viewing data and different features
train.head()


# In[ ]:


train.shape 


# In[ ]:


train.columns 


# In[ ]:


train['Sex'].value_counts()


# # Data Visualization

# In[ ]:


#Visualizing survivals based on gender
train['Died'] = 1 - train['Survived']
train.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar',
                                                           figsize=(10, 5),
                                                           stacked=True)


# In[ ]:


##Visualizing survivals based on fare
figure = plt.figure(figsize=(16, 7))
plt.hist([train[train['Survived'] == 1]['Fare'], train[train['Survived'] == 0]['Fare']], 
         stacked=True, bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()


# # Processing training data

# In[ ]:


#Cleaning the data by removing irrelevant columns
df1=train.drop(['Name','Ticket','Cabin','PassengerId','Died'], axis=1)
df1.head(10)


# In[ ]:


df1.isnull().sum() 


# In[ ]:


#Converting the categorical features 'Sex' and 'Embarked' into numerical values 0 & 1
df1.Sex=df1.Sex.map({'female':0, 'male':1})
df1.Embarked=df1.Embarked.map({'S':0, 'C':1, 'Q':2,'nan':'NaN'})
df1.head()


# In[ ]:


#Mean age of each sex
mean_age_men=df1[df1['Sex']==1]['Age'].mean()
mean_age_women=df1[df1['Sex']==0]['Age'].mean()


# In[ ]:


#Filling all the null values in 'Age' with respective mean age
df1.loc[(df1.Age.isnull()) & (df1['Sex']==0),'Age']=mean_age_women
df1.loc[(df1.Age.isnull()) & (df1['Sex']==1),'Age']=mean_age_men


# In[ ]:


#Let's check for the null values again now
df1.isnull().sum()


# In[ ]:


#Since there exist 2 null values in the Embarked column, let's drop those rows containing null values
df1.dropna(inplace=True)


# In[ ]:


df1.isnull().sum()


# In[ ]:


#Doing Feature Scaling to standardize the independent features present in the data in a fixed range
df1.Age = (df1.Age-min(df1.Age))/(max(df1.Age)-min(df1.Age))
df1.Fare = (df1.Fare-min(df1.Fare))/(max(df1.Fare)-min(df1.Fare))
df1.describe()


# # Creating model

# In[ ]:


#Splitting the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df1.drop(['Survived'], axis=1),
    df1.Survived,
    test_size= 0.2,
    random_state=0,
    stratify=df1.Survived)


# **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
lrmod = LogisticRegression()