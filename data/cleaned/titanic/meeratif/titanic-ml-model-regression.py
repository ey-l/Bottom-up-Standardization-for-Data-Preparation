#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


train_file = pd.read_csv('data/input/titanic/train.csv')
test_file = pd.read_csv('data/input/titanic/test.csv')
sub_file = pd.read_csv('data/input/titanic/gender_submission.csv')
sub_file_1 = pd.read_csv('data/input/titanic/gender_submission.csv')
sub_file_2 = pd.read_csv('data/input/titanic/gender_submission.csv')
sub_file_4 = pd.read_csv('data/input/titanic/gender_submission.csv')
sub_file_5 = pd.read_csv('data/input/titanic/gender_submission.csv')


# In[ ]:


train_file.shape


# In[ ]:


train_file.dtypes


# In[ ]:


train_file.info()


# In[ ]:


train_file.columns


# In[ ]:


train_file.head()


# In[ ]:


train_file.tail()


# In[ ]:


plt.figure(figsize=(4,4))
plt.bar(list(train_file['Survived'].value_counts().keys()), list(train_file['Survived'].value_counts()), color=('grey', 'silver'))
plt.title("Num of Survived People")



# In[ ]:


train_file['Survived'].value_counts()


# In[ ]:


sum(train_file['Survived'].isnull())


# In[ ]:


plt.figure(figsize=(5,5))
plt.bar(list(train_file['Pclass'].value_counts().keys()), list(train_file['Pclass'].value_counts()))
plt.title("Passengers CLass")
plt.xlabel("Classes")
plt.ylabel("Num of Passengers")



# In[ ]:


train_file['Pclass'].value_counts() #Passengers Class


# In[ ]:


plt.figure(figsize=(5,5))
plt.bar(list(train_file['Sex'].value_counts().keys()), list(train_file['Sex'].value_counts()))
plt.title("Gender")
plt.ylabel("Num of Passengers")



# In[ ]:


train_file['Sex'].value_counts()


# In[ ]:


#Historgram
plt.figure(figsize=(5,5))
plt.hist(train_file['Age'])
plt.title("AGE")


#Bar
plt.figure(figsize=(5,5))
plt.bar(list(train_file['Age'].value_counts().keys()), list(train_file['Age'].value_counts()))
plt.title("Age")
plt.ylabel("Num of Passengers")



# In[ ]:


sns.countplot(x="Survived", hue='Sex', data=train_file, palette="winter")


# In[ ]:


sns.boxplot(x="Pclass", y="Age", data=train_file)


# In[ ]:


sns.boxplot(x="Sex", y="Age", data=train_file)


# In[ ]:


train_file.isnull().sum()


# In[ ]:


sum(train_file['Age'].isnull())


# In[ ]:


pd.get_dummies(train_file['Sex'])


# In[ ]:


train_file_update = train_file.dropna() # **************** Remove all null values from dataset


# In[ ]:


plt.scatter(train_file_update['Age'], train_file_update['Survived'], marker="*")


# In[ ]:


train_file_update = train_file.dropna() # **************** Remove all null values from dataset


# In[ ]:


print(sum(train_file_update['Survived'].isnull()))
print(sum(train_file_update['Age'].isnull()))


# In[ ]:


# Dependent Variable is survived
# Independent Variable is Age
x_train_data = train_file_update['Age']
y_train_data = train_file_update['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(train_file_update[['Age']], train_file_update.Survived, test_size=0.1)


# In[ ]:


print(sum(y_test), sum(y_train))


# In[ ]:


x_train


# # LogisticRegression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# importing Lib for Prediction 
# Using Logistics Regression

# In[ ]:


model = LogisticRegression()


# In[ ]:

