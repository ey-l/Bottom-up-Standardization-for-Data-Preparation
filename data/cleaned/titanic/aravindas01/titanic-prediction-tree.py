#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# reading train data
train=pd.read_csv('data/input/titanic/train.csv')

# reading test data
test=pd.read_csv('data/input/titanic/test.csv')


# In[ ]:


# printing first five rows of the data set

train.head()


# In[ ]:


# number of rows and columns
train.shape 


# In[ ]:


# column names
train.columns 


# In[ ]:


# number of null values in dataset
train.isnull().sum() 


# In[ ]:


train['Sex'].value_counts()


# In[ ]:


#countplot
sns.countplot(x='Sex', data=train)


# In[ ]:


train['Pclass'].value_counts()


# In[ ]:


sns.countplot(x='Pclass', data=train)


# In[ ]:


train['Embarked'].value_counts()


# In[ ]:


sns.countplot(x='Embarked', data=train)


# In[ ]:


train['SibSp'].value_counts()


# In[ ]:


sns.countplot(x='SibSp', data=train)


# In[ ]:


train['Died'] = 1 - train['Survived']


# In[ ]:


#Visualisating survival based on gender


train.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar',
                                                           figsize=(8, 5),
                                                           stacked=True)


# In[ ]:


train.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar',
                                                            figsize=(10, 5),
                                                            stacked=True)


# In[ ]:


#visualizing survival based on the fare

figure = plt.figure(figsize=(14, 7))
plt.hist([train[train['Survived'] == 1]['Fare'], train[train['Survived'] == 0]['Fare']], 
         stacked=True, bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()


# In[ ]:


#Cleaning the train dataset by dropping unwanted columns

df1=train.drop(['Name','Ticket','Cabin','PassengerId','Died'], axis=1)
df1.head(10)


# In[ ]:


# Converting categorical feature to numeric

df1.Sex=df1.Sex.map({'female':0, 'male':1})
df1.Embarked=df1.Embarked.map({'S':0, 'C':1, 'Q':2,'nan':'NaN'})
df1.head()


# In[ ]:


# median age of each sex

median_age_men=df1[df1['Sex']==1]['Age'].median()
median_age_women=df1[df1['Sex']==0]['Age'].median()


# In[ ]:


# fill the null values in 'Age' with respective median age

df1.loc[(df1.Age.isnull()) & (df1['Sex']==1),'Age']=median_age_men
df1.loc[(df1.Age.isnull()) & (df1['Sex']==0),'Age']=median_age_women


# In[ ]:


# checking for null values

df1.isnull().sum()


# In[ ]:


# dropping two rows with null value

df1.dropna(inplace=True)


# In[ ]:


# Checking again for null values

df1.isnull().sum()


# In[ ]:


# cleaned dataset

df1.head()


# In[ ]:


#Feature Scaling

df1.Age = (df1.Age-min(df1.Age))/(max(df1.Age)-min(df1.Age))
df1.Fare = (df1.Fare-min(df1.Fare))/(max(df1.Fare)-min(df1.Fare))

df1.describe()


# ## Data Modelling

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    df1.drop(['Survived'], axis=1),
    df1.Survived,
    test_size= 0.2,
    random_state=0,
    stratify=df1.Survived)


# In[ ]:


# Logistic regression

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()