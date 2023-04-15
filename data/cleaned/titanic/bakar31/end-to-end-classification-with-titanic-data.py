#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# In[ ]:


train = pd.read_csv('data/input/titanic/train.csv')
test = pd.read_csv('data/input/titanic/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.shape, test.shape


# ## Data Preprocessing on Train data
# 
# Removing columns that we don't need

# In[ ]:


train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)


# Checking for null values

# In[ ]:


train.isna().sum()


# We have null values in two columns. Let's take care of this problem.
# 
# Let's fill the null values of age columns with the mean values

# In[ ]:


train['Age'].fillna(train['Age'].mean(), inplace = True)


# Now we have to take care of null values of Embarked column.
# 
# Let's first check which embarkation port we have most in our dataset.

# In[ ]:


train.Embarked.value_counts()


# `Southampton` is the top port of embarkation. So, let's fill the null values with `S`

# In[ ]:


train['Embarked'].fillna('S', inplace = True)


# Let's check again for null values.

# In[ ]:


train.isna().sum()


# **Nice!**
# 
# We don't any null values now

# # Data Exploration on Train set

# Let's first check how many people survived

# In[ ]:


train.Survived.value_counts()


# In[ ]:


train.Survived.value_counts().plot(kind = 'bar', color = ['lightblue', 'lightgreen']);


# Let's check how many male and female was there

# In[ ]:


train.Sex.value_counts()


# In[ ]:


train.Sex.value_counts().plot(kind = 'bar', color = ['skyblue', 'plum']);


# let's check out survivors w.r.t sex

# In[ ]:


pd.crosstab(train.Sex, train.Survived)


# In[ ]:


pd.crosstab(train.Sex, train.Survived).plot(kind = 'bar', color = ['slategray', 'salmon']);


# Survivors w.r.t pclass

# In[ ]:


pd.crosstab(train.Pclass, train.Survived)


# In[ ]:


pd.crosstab(train.Pclass, train.Survived).plot(kind = 'bar', color = ['slategray', 'lightcoral']);


# Let's check the Port of Embarkation

# In[ ]:


train.Embarked.value_counts()


# Let's look at our age column

# In[ ]:


sns.countplot(x = 'Embarked', data = train);


# In[ ]:


sns.displot(x = 'Age', data = train, color = 'cadetblue', kde = True);


# In[ ]:


sns.displot(x = 'Fare', data = train, kind = 'kde');


# Let's now find a relation among age, survived and pclass columns

# In[ ]:


sns.lmplot(x = 'Age', y = 'Survived', hue = 'Pclass', data = train);


# In[ ]:


correlation_matrix = train.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, 
            annot=True, 
            linewidths=0.5, 
            fmt= ".2f", 
            cmap="YlGnBu");


# # Feature Engineering in train data

# In[ ]:


train['family'] = train['SibSp'] + train['Parch']


# In[ ]:


train.head(10)


# Removing skewness in `Age` column

# In[ ]:


train['Age']=np.log(train['Age']+1)


# In[ ]:


train['Age'].plot(kind = 'density', figsize=(10, 6));


# Removing skewness in `Fare` column

# In[ ]:


train['Fare']=np.log(train['Fare']+1)


# In[ ]:


train['Fare'].plot(kind = 'density', figsize=(10, 6));


# In[ ]:


train.head(10)


# Let's create x and y matrix of features

# In[ ]:


x = train.drop('Survived',  axis = 1)
y = train['Survived']


# In[ ]:


x.shape


# In[ ]:


x.head()


# We have two `categorical` columns. Let's take care of them now.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ['Sex', 'Embarked', 'Pclass']
onehotencode = OneHotEncoder()

transformer = ColumnTransformer([('Encoder', onehotencode, categorical_features)], remainder = 'passthrough')

encoded = transformer.fit_transform(x)


# In[ ]:


encoded_df = pd.DataFrame(encoded)


# In[ ]:


encoded_df.shape


# In[ ]:


encoded_df.head()


# **Avoiding Dummy variables**

# In[ ]:


encoded_x = encoded_df.drop([0, 2, 5], axis = 1)


# In[ ]:


encoded_x.head()


# In[ ]:


encoded_x.shape


# In[ ]:


y.shape


# # Feature Engineering in test data

# In[ ]:


test['family'] = test['SibSp'] + test['Parch']


# In[ ]:


test.head()


# Removing skewness in `Age` column

# In[ ]:


test['Age']=np.log(test['Age']+1)


# Removing skewness in `Fare` column

# In[ ]:


test['Fare']=np.log(test['Fare']+1)


# In[ ]:


test['Age'].plot(kind = 'density', figsize=(10, 6));


# In[ ]:


test['Fare'].plot(kind = 'density', figsize=(10, 6));


# In[ ]:


test.head(10)


# # Preparing test set

# In[ ]:


test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)


# In[ ]:


test.head(10)


# Checking for null values

# In[ ]:


test.isna().sum()


# In[ ]:


test['Age'].fillna(test['Age'].mean(), inplace = True)
test['Fare'].fillna(test['Fare'].mean(), inplace = True)


# In[ ]:


test.isna().sum()


# We succesfully removed all the null values

# As before we now have to take care of `categorical columns`

# In[ ]:


categorical_features = ['Sex', 'Embarked', 'Pclass']
onehotencode = OneHotEncoder()

transformer = ColumnTransformer([('Encoder', onehotencode, categorical_features)], remainder = 'passthrough')

encoded_test = transformer.fit_transform(test)


# In[ ]:


encoded_test = pd.DataFrame(encoded_test)


# In[ ]:


encoded_test.head()


# Avoiding dummy variable trap

# In[ ]:


encoded_test_x = encoded_test.drop([0, 2, 5], axis = 1)


# In[ ]:


encoded_test_x.head()


# In[ ]:


encoded_test_x.shape


# # Modeling

# Let's split our dataset

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(encoded_x,y,random_state = 31)


# In[ ]:


len(x_train), len(x_test), len(y_train), len(y_test)


# In[ ]:


x_train.shape


# In[ ]:


y_train.shape


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression(max_iter = 1000, random_state = 4)