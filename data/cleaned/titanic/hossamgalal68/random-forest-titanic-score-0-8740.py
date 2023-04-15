#!/usr/bin/env python
# coding: utf-8

# # Titanic - Machine Learning
# ## Overview :
# ### 1- Import Libraries
# ### 2- Read Data
# ### 3- Data Cleaning
# ### 4- Data Encoding 
# ### 5- EDA
# ### 6- Data Splitting
# ### 7- Data Scalling
# ### 8- Model Building
# 
# #### By : Hossam Galal

# --------------------
# # Import Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer as si
from sklearn.preprocessing import OneHotEncoder , LabelEncoder , MinMaxScaler , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


# -----------------
# # Read Data

# In[ ]:


training = pd.read_csv('data/input/titanic/train.csv')
test = pd.read_csv("data/input/titanic/test.csv")
sub = pd.read_csv('data/input/titanic/gender_submission.csv')


# In[ ]:


all_data = pd.concat([training,test])


# In[ ]:


print ('All data shape is ', all_data.shape )
print ('Training data shape is ', training.shape )
print ('Test data shape is ', test.shape )
print ('Submission data shape is ', sub.shape )


# In[ ]:


training


# In[ ]:


training.info()


# we have 11 columns in train data --> 5 columns are object data type ---> we have 3 columns have null values

# In[ ]:


training.describe()


# In[ ]:


test.info()


# we have 10 columns in test data --> 5 columns are object data type ---> we have 3 columns have null values

# In[ ]:


test.describe()


# --------------
# # Data Cleaning
# ### clean training data

# In[ ]:


training.isnull().sum()


# column of 'cabin' has 687 null values so it not important (we will drop it)
# 
# column of 'Age' has 177 null values(not large number) so we will try filling the null values 
# 
# column of 'Embarked' has 2 null values so we will drop the 2 rows which have null values

# In[ ]:


training['Age'] = training['Age'].fillna(training['Age'].mean())
training=training.drop(['Cabin', 'Ticket','Name'],1)
training= training.dropna(axis=0, subset=['Embarked']) 
training.fillna(0, inplace=True)


# we delete 'Ticket','Name' columns becouse them are not emportant

# In[ ]:


training


# In[ ]:


training.isnull().sum()


# now, training data is clean

# ### Clean test data

# In[ ]:


test


# In[ ]:


test.isnull().sum()


# column of 'cabin' has 327 null values so it not important (we will drop it)
# 
# column of 'Age' has 86 null values(not large number) so we will try filling the null values 
# 
# column of 'Fare' has 1 null value so we will drop the 1 rows which have null values

# In[ ]:


test['Age'] = test['Age'].fillna(test['Age'].mean())
test=test.drop(['Cabin', 'Ticket','Name'],1)
test.fillna(0, inplace=True)


# we delete 'Ticket','Name' columns becouse them are not emportant

# In[ ]:


test.isnull().sum()


# now, test data is clean

# In[ ]:


test


# ------------
# # Data Encoding
# ### let's start with training data

# In[ ]:


training_encoding = training.copy()


# In[ ]:


training_encoding


# In[ ]:


training_encoding.info()


# we have 2 columns need encoding (Sex,Embarked)

# In[ ]:


training_encoding['Sex'].value_counts()


# In[ ]:


training_encoding['Sex'] = training_encoding['Sex'].factorize(['female','male'])[0]


# In[ ]:


training_encoding['Sex'].value_counts()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
training_encoding['Embarked_N']=label_encoder.fit_transform(training_encoding['Embarked'])
training_encoding


# In[ ]:


training_encoding.drop('Embarked', inplace=True, axis=1)
training_encoding


# In[ ]:


training_encoding.info()


# now, train data is encoded

# ### Test data Encoding

# In[ ]:


test_encoding = test.copy()


# In[ ]:


test_encoding.info()


# we have 2 columns need encoding (Sex,Embarked)

# In[ ]:


test_encoding['Sex'].value_counts()


# In[ ]:


test_encoding['Sex'] = test_encoding['Sex'].factorize(['female','male'])[0]
test_encoding['Sex'].value_counts()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
test_encoding['Embarked_N']=label_encoder.fit_transform(test_encoding['Embarked'])
test_encoding


# In[ ]:


test_encoding.drop('Embarked', inplace=True, axis=1)
test_encoding


# In[ ]:


test_encoding.info()


# now, test data is encoded

# ------------
# #### we should make concatination between test data and submission

# In[ ]:


df_test=test_encoding.copy()


# In[ ]:


df_test['Survived'] = sub['Survived']


# In[ ]:


df_test


# In[ ]:


df_test.describe()


# In[ ]:


df_test.isnull().sum()


# --------------
# # EDA

# In[ ]:


df_training=training_encoding.copy()


# In[ ]:


df_training


# In[ ]:


plt.figure(figsize=(30, 30))
sns.heatmap(df_training.corr(), annot=True, cmap="mako", annot_kws={"size":14})


# In[ ]:


fig = plt.figure(figsize=(5,5))
df_training['Survived'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Survived")
print("")


# number of survived people less than number of not survived people

# In[ ]:


sns.countplot(data=df_training, x = 'Survived', hue='Pclass').set(xticklabels = ['Did not Survive', 'Survied'], title = 'Titanic Survival Data')



# people that have pclass=3 are have highest number of not survive so class1 and class2 more secure than class3

# In[ ]:


sns.countplot(data=df_training, x = 'Survived', hue='Sex').set(xticklabels = ['Did not Survive', 'Survied'], title = 'Titanic Survival Data')



# male is 1 , female is 0
# 
# so number of men (not survive) > number of women (not survive)

# In[ ]:


sns.distplot(df_training['Age'])



# In[ ]:


fig = plt.figure(figsize=(5,5))
df_training['Embarked_N'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Embarked_N")
print("")


# C=0
# Q=1
# S=2

# In[ ]:


sns.countplot(data=df_training, x = 'Survived', hue='SibSp').set(xticklabels = ['Did not Survive', 'Survied'], title = 'Titanic Survival Data')



# In[ ]:


sns.countplot(data=df_training, x = 'Survived', hue='Parch').set(xticklabels = ['Did not Survive', 'Survied'], title = 'Titanic Survival Data')



# --------------------
# # Data Splitting

# In[ ]:


train_split=df_training.copy()
test_split=df_test.copy()


# In[ ]:


train_split


# In[ ]:


test_split


# In[ ]:


train_split.columns


# In[ ]:


test_split.columns


# In[ ]:


train_split=train_split[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_N','Survived']]


# In[ ]:


test_split=test_split[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_N','Survived']]


# In[ ]:


x_train = train_split.drop(["Survived"],axis=1).values
y_train= train_split['Survived'].values


# In[ ]:


x_test = test_split.drop(["Survived"],axis=1).values
y_test= test_split['Survived'].values


# ---------
# # Data Scalling

# In[ ]:


scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)


# In[ ]:


scalar=StandardScaler()
x_test=scalar.fit_transform(x_test)


# ---------
# # Random Forest Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


rf= RandomForestClassifier(n_estimators=100,max_leaf_nodes=20)