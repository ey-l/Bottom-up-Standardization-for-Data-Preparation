#!/usr/bin/env python
# coding: utf-8

# # Kaggle - Titanic: Machine Learning from Disaster
# 
# The competition details as below
# 
# ### The Challenge
# 
# The sinking of the Titanic is one of the most infamous shipwrecks in history.
# 
# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# 
# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# 
# In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc). 
# 
# ### Predicting the survival on the Titanic

# **Prediction Results : 0.78947**

# ### Load Helpful Packages

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# ### Load the Data 

# In[ ]:


train_data = pd.read_csv('data/input/titanic/train.csv')
train_data.head(10)


# In[ ]:


test_data = pd.read_csv('data/input/titanic/test.csv')
test_data.head(10)


# In[ ]:


train_data.shape


# In[ ]:


test_data.shape


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# ### Check Missing Values 

# In[ ]:


# Checking Missing values in train_data
train_data.isnull().sum()


# In[ ]:


# Checking Missing values in test_data
test_data.isnull().sum()


# In[ ]:


train_data.columns


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


sns.catplot(x = 'Embarked', kind = 'count', data = train_data)


# In[ ]:


train_data['Embarked'] = train_data['Embarked'].fillna("S")


# In[ ]:


train_data.isnull().sum()


# In[ ]:


train_data['Cabin'] = train_data['Cabin'].fillna("Missing")
test_data['Cabin'] = test_data['Cabin'].fillna("Missing")


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


test_data['Fare'] = test_data['Fare'].median()


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# #### No missing values left so we can proceed further

# In[ ]:


## get dummy variables for Column sex and embarked since they are categorical value.
train_data = pd.get_dummies(train_data, columns=["Sex"], drop_first=True)
train_data = pd.get_dummies(train_data, columns=["Embarked"],drop_first=True)


#Mapping the data.
train_data['Fare'] = train_data['Fare'].astype(int)
train_data.loc[train_data.Fare<=7.91,'Fare']=0
train_data.loc[(train_data.Fare>7.91) &(train_data.Fare<=14.454),'Fare']=1
train_data.loc[(train_data.Fare>14.454)&(train_data.Fare<=31),'Fare']=2
train_data.loc[(train_data.Fare>31),'Fare']=3

train_data['Age']=train_data['Age'].astype(int)
train_data.loc[ train_data['Age'] <= 16, 'Age']= 0
train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 32), 'Age'] = 1
train_data.loc[(train_data['Age'] > 32) & (train_data['Age'] <= 48), 'Age'] = 2
train_data.loc[(train_data['Age'] > 48) & (train_data['Age'] <= 64), 'Age'] = 3
train_data.loc[train_data['Age'] > 64, 'Age'] = 4


# In[ ]:


## get dummy variables for Column sex and embarked since they are categorical value.
test_data = pd.get_dummies(test_data, columns=["Sex"], drop_first=True)
test_data = pd.get_dummies(test_data, columns=["Embarked"],drop_first=True)


#Mapping the data.
test_data['Fare'] = test_data['Fare'].astype(int)
test_data.loc[test_data.Fare<=7.91,'Fare']=0
test_data.loc[(test_data.Fare>7.91) &(test_data.Fare<=14.454),'Fare']=1
test_data.loc[(test_data.Fare>14.454)&(test_data.Fare<=31),'Fare']=2
test_data.loc[(test_data.Fare>31),'Fare']=3

test_data['Age']=test_data['Age'].astype(int)
test_data.loc[ test_data['Age'] <= 16, 'Age']= 0
test_data.loc[(test_data['Age'] > 16) & (test_data['Age'] <= 32), 'Age'] = 1
test_data.loc[(test_data['Age'] > 32) & (test_data['Age'] <= 48), 'Age'] = 2
test_data.loc[(test_data['Age'] > 48) & (test_data['Age'] <= 64), 'Age'] = 3
test_data.loc[test_data['Age'] > 64, 'Age'] = 4


# In[ ]:


# In our data the Ticket and Cabin,Name are the base less,leds to the false prediction so Drop both of them.
train_data.drop(['Ticket','Cabin','Name'],axis=1,inplace=True)
test_data.drop(['Ticket','Cabin','Name'],axis=1,inplace=True)


# ## Exploratory Data Analysis 

# In[ ]:


train_data.describe()


# In[ ]:


train_data.Survived.value_counts()/len(train_data)*100
#This signifies almost 61% people in the ship died and 38% survived.


# In[ ]:


train_data.groupby("Survived").mean()


# In[ ]:


train_data.groupby("Sex_male").mean()


#  #### The points to know from the analysis
#  #### 1. 38% of people survived
#  #### 2. 74% of Females survived and ~19% of Males survived 

# ### Correlation between Variables

# In[ ]:


train_data.corr()


# In[ ]:


#Heatmap
plt.subplots(figsize=(10,8))
sns.heatmap(train_data.corr(),annot=True,cmap='Blues_r')
plt.title("Correlation Among Variables", fontsize = 20);


# - Survived has positive correlation of 0.3 with Fare
# - Sex and survived have negative correlation of -0.54
# - Pclass and Survived have negative correlation of -0.34**

# In[ ]:


sns.barplot(x="Sex_male",y="Survived",data=train_data)
plt.title("Gender Distribution - Survived", fontsize = 16)


# ##### Female passengers have survived more than male passengers i.e Females and Children would have been the priority

# In[ ]:


sns.barplot(x='Pclass',y='Survived',data=train_data)
plt.title("Passenger Class Distribution - Survived", fontsize = 16)


# ### Survival as per classes
# - 63% of Passenger Class 1
# - 48% of Passenger Class 2
# - Only 25% of Passenger Class 3 survived

# ### Modeling Data 
# ###### I will be modelling the data with the below models:
# - Logistic Regression
# - Support Vector Machine
# - Decision Tree Classifier
# - Random Forest Classifier
# - K-Nearest Neighbour Classifier
# - Gradient Boosting
# - Grid SearchCV

# In[ ]:


from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[ ]:


X = train_data.drop(['Survived'], axis=1)
y = train_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state = 5)


# In[ ]:


print(len(X_train),len(X_test),len(y_train),len(y_test))


# ### Logistic Regression

# In[ ]:


#Logistic Regression
logReg = LogisticRegression()