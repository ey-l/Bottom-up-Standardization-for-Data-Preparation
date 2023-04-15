#!/usr/bin/env python
# coding: utf-8

# ## Welcome!
# **Suggesstion, if any, are most welcomed. I would love hearing from you & improving myself. Please consider providing your opinion.**
# 
# * Model Used: Support Vector Machine with Gaussian Kernel
# * Run Time: 22 sec (may vary in different verions)
# * Score: 0.77990
# 
# *This notebook is for beginnner made by a beginner. If you're unable to understand anything leave a comment.*

# ## Importing Libraries

# In[ ]:


# We're using support vector machine model here.

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler 


# ## Importing & visualizing data

# **Importing Data**

# In[ ]:


test_data = pd.read_csv('data/input/titanic/test.csv')
train_data = pd.read_csv('data/input/titanic/train.csv')
sub_data = pd.read_csv('data/input/titanic/gender_submission.csv')


# **VISUALIZING TRAINING DATASET**

# In[ ]:


train_data


# **Various columns in dataset are:**
# 1. **PassengerId:** Unique Id for each passenger
# 1. **Survived:** 0 = No, 1 = Yes
# 1. **Pclass:** Ticket class 
# 1. **Sex:** Gender 
# 1. **Age:** Age of passenger in years
# 1. **SibSp:** Number of siblings/spouses alongwith
# 1. **Parch:** Number of parents/children alongwith
# 1. **Ticket:** Ticket number
# 1. **Fare:** Ticket price
# 1. **Cabin:** Cabin number
# 1. **Embarked:** Embarkation Port (3 entries C, Q, S)

# In[ ]:


train_data.info()  #for checking null values & data type


# In[ ]:


test_data.describe() #for gettig some more insights


# In[ ]:


print('Number of null values in different columns are: ')
print('--------------------------------------------------')
print(train_data.isna().sum())
print('--------------------------------------------------')


# ## Data Preprocessing

# **Dealing with Cabin Column**
# 
# In the Cabin column, 687 out of 891 entries are missing. So, here either we should drop this column or assign 1 to that column having Cabin value & 0 that misses the value. Maybe that give some information. Who knows!
# 
# Lets try it...

# In[ ]:


train_data.loc[(train_data['Cabin'].isna() == False), 'Cabin'] = 1
train_data.loc[(train_data['Cabin'].isna() == True), 'Cabin'] = 0


# **Filling Missing Values in Age Column**
# 
# Lets 1st keep it simple! Just take average age & put it at missing values.

# In[ ]:


train_data.loc[(train_data['Age'].isna() == True), 'Age'] = train_data['Age'].mean()


# **Filling Missing Values in Embarked Column**
# 
# There're only 2 values missing. So lets assume those values to be equal to the mode of this column.
# 

# In[ ]:


train_data['Embarked'].mode()


# In[ ]:


train_data.loc[(train_data['Embarked'].isna() == True), 'Embarked'] = 'S'


# **Just re-check for missing values**

# In[ ]:


print('Number of null values in different columns are: ')
print('--------------------------------------------------')
print(train_data.isna().sum())
print('--------------------------------------------------')


# **Drop some columns**
# 
# Lets drop some of the columns which doesnt't seem to be relevant. Here we go!

# In[ ]:


train_data.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)


# **Make all columns to have numerical values**
# 
# * Put 1 for female & 0 for male in 'Sex' Column
# * Use one-hot encoding for 'Embarked' Column

# In[ ]:


train_data.loc[(train_data['Sex'] == 'female'), 'Sex'] = 1
train_data.loc[(train_data['Sex'] == 'male'), 'Sex'] = 0


# In[ ]:


ohe = OneHotEncoder(sparse=False, handle_unknown='error', drop='first')
ohe_df = pd.DataFrame(ohe.fit_transform(train_data[['Embarked']]))

ohe_df.columns = ohe.get_feature_names(['Embarked'])

ohe_df.head()


# In[ ]:


train_data = pd.concat([train_data,ohe_df], axis=1)
train_data.drop(['Embarked'], axis = 1, inplace = True)


# **Time to scale the Data**
# 
# Alright! Don't you think its a good time to scale the data. Lets do it...

# In[ ]:


ColumnToScale = ['Age', 'Fare']      #(Rest others are almost on same scale)

train_data[ColumnToScale] = MinMaxScaler().fit_transform(train_data[ColumnToScale])

train_data['Fare'] = train_data['Fare'] * 10     #as most of values in Fare became negligible after scaling


# In[ ]:


train_data #Whoa! We changed it...


# In[ ]:


train_data.describe()


# ## Model Applying
# 
# Now everything seems good.
# Lets apply our learning model. We're using **SVM with Gaussian Kernel** here.

# In[ ]:


train_data.shape


# In[ ]:


X_train = train_data.iloc[:,1:10].values
Y_train = train_data.iloc[:,0].values


# In[ ]:


classifier = SVC(kernel='rbf', random_state = 1)