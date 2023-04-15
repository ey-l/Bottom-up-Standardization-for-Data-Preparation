#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# In[ ]:


#Loading data
titanic = pd.read_csv('data/input/titanic/train.csv')
titanic.head() #returns top 5 rows


# In[ ]:


titanic.shape


# In[ ]:


#visualizing the data


# In[ ]:


sns.countplot(x = 'Survived', data = titanic)


# We can observe that those who are not survived are greater than those who survived.

# In[ ]:


sns.countplot(x = 'Survived', hue = 'Sex', data = titanic, palette = 'ocean')


# here 0 represents not survived and 1 represents survived.
# here females are more likely to survive.

# In[ ]:


sns.countplot(x = 'Survived', hue = 'Pclass', data = titanic, palette = 'flare')


# The passengers who doesnot survive belongs to 3rd class.
# 1st class passengers are more likely to survive.

# In[ ]:


titanic['Age'].plot.hist()


# highest number of travellers belong to the age group 20-40.
# only few people belong to age group 70-80

# In[ ]:


titanic['Fare'].plot.hist(bins=20,figsize=(10,5))


# We can observe that most of the tickets bought are under fare 100.
# Very few are on the range 200-500

# In[ ]:


sns.countplot(x = 'SibSp',data = titanic, palette = 'rocket')


# We can observe that most of the passengers doesnot have siblings aboard.

# In[ ]:


titanic['Parch'].plot.hist()


# In[ ]:


sns.countplot(x = 'Parch',data = titanic, palette = 'magma')


# The numbers of parents and siblings who aboard the ship are less.

# In[ ]:


titanic.head()


# In[ ]:


#deleting unwanted columns
titanic.drop(['Name','PassengerId','Ticket','Cabin'], axis=1, inplace = True)


# In[ ]:


titanic.head()


# In[ ]:


titanic.shape #returns no of rows and columns


# In[ ]:


#coverting into categorical data


# In[ ]:


titanic=pd.get_dummies(titanic)
titanic.head()


# In[ ]:


titanic['Age']=titanic['Age']/max(titanic['Age'])
titanic['Fare']=titanic['Fare']/max(titanic['Fare'])


# In[ ]:


titanic.head()


# In[ ]:


#checking null values
titanic.isnull().sum()


# In[ ]:


#removing null values
titanic['Age'].fillna(titanic['Age'].median(),inplace = True)


# In[ ]:


titanic.isnull().sum()


# In[ ]:


titanic.head()


# In[ ]:


titanic.shape


# In[ ]:


#training the data


# In[ ]:


x = titanic.drop('Survived', axis = 1)
y = titanic['Survived']


# In[ ]:


#train-test split


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 30, random_state = 42)


# In[ ]:


#logistic regression


# In[ ]:


from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()


# In[ ]:

