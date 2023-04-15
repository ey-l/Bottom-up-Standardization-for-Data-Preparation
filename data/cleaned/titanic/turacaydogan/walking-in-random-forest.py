#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB


# Let's read the training set

# In[ ]:


train = pd.read_csv('data/input/titanic/train.csv')
train.describe()


# Women and children first..

# In[ ]:


maleProb=train.Survived.loc[train.Sex=='male' ].mean()
print('male surviving probability:',maleProb)

femaleProb=train.Survived.loc[train.Sex=='female' ].mean()
print('female surviving probability:',femaleProb)


# In[ ]:


sns.catplot(x="Sex", y="Age", hue="Survived", kind="swarm", data=train)


# Life is pay to win, poor folx could not make it

# In[ ]:


sns.catplot(x="Pclass", y="Survived", kind="bar", data=train)


# If you are a man and you are rich, you almost have the same chance of surviving with a poor female.

# In[ ]:


sns.catplot(x="Sex", y="Survived", hue='Pclass', kind="bar", data=train)


# If you have a parent or a child, you have a better chance to live because most children survived. 
# Further, if you have a lover or a sibling, you have something to live for or you are just saved by your lover while they are food to the sharx. 
# Being a loner in a ship is the worst. Not if you have fat stax of money, tho.

# In[ ]:


Families=train.Survived.loc[(train.Parch>0)].mean()
print('Parents&Childs:',Families)
Lovers=train.Survived.loc[(train.Parch==0)&(train.SibSp==1)].mean()
print('Lovers:',Lovers)
Loners=train.Survived.loc[(train.Parch==0)].mean()
print('Loners:',Loners)
totalLoners=train.Survived.loc[(train.Parch==0)&(train.SibSp==0)].mean()
print('Total Loners:',totalLoners)
richLoners=train.Survived.loc[(train.Parch==0)&(train.SibSp==0)&(train.Pclass==1)].mean()
print('Rich Loners:',richLoners)


# In[ ]:


train_x=train[['Age','Sex','Parch','Pclass','SibSp']]
print(train_x.dtypes)
train_y=train[['Survived']]

#sex is object type, turned into integer
train_x.Sex.loc[train_x.Sex=='male']=0
train_x.Sex.loc[train_x.Sex=='female']=1
train_x.Sex=train_x['Sex'].astype('str').astype(int)

print(train_x.dtypes)


# See if there is any missing value

# In[ ]:


missingColumns_x = [col for col in train_x.columns
                     if train_x[col].isnull().any()]
print(missingColumns_x)
missingColumns_y = train_y.isnull().any()
print(missingColumns_y)


# I am going to use age column in my model, so I will impute the missing values using KNN imputer. This imputer treats nearest neighboring values to make a prediction of the missing value. Simple imputers generally use central distribution methods such as mean, mode and median.

# In[ ]:


knn_imputer = KNNImputer(n_neighbors=4, weights="uniform")
train_xi=pd.DataFrame(knn_imputer.fit_transform(train_x))


# Before I start with my model I want to inspect test data as well.

# In[ ]:


test = pd.read_csv('data/input/titanic/test.csv')
test.describe()

test_x=test[['Age','Sex','Parch','Pclass','SibSp']]
#test_y=test[['Survived']]

#sex is object type turned into integer
test_x.Sex.loc[test_x.Sex=='male']=0
test_x.Sex.loc[test_x.Sex=='female']=1
test_x.Sex=test_x['Sex'].astype('str').astype(int)

print(train_x.dtypes)


# In[ ]:


missingColumns_x = [col for col in test_x.columns
                     if test_x[col].isnull().any()]
print(missingColumns_x)

knn_imputer = KNNImputer(n_neighbors=4, weights="uniform")
test_xi=pd.DataFrame(knn_imputer.fit_transform(test_x))


# I can start my model with a random forest

# In[ ]:


rfmodel=RandomForestClassifier(random_state=1)