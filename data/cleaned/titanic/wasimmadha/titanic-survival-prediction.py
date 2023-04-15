#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Prediction
# 
# The Titanic Ship was the largest and the  luxurious ship in the world. But unfortunately, it sunk on its maiden voyage. While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


test_df = pd.read_csv('data/input/titanic/test.csv')
train_df = pd.read_csv('data/input/titanic/train.csv')
submission = pd.read_csv('data/input/titanic/gender_submission.csv')


# # Data Analysis and Feature Engineering

# In[ ]:


test_df.head()


# In[ ]:


train_df.head()


# In[ ]:


train_df = pd.get_dummies(train_df, columns=['Embarked'])


# In[ ]:


train_df.head()


# In[ ]:


test_df = pd.get_dummies(test_df, columns=['Embarked'])


# In[ ]:


test_df.head()


# In[ ]:


train_df.head()


# In[ ]:


train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)


# In[ ]:


test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)


# In[ ]:


train_df['family_size'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df.head()


# In[ ]:


train_df[['family_size', 'Survived']].groupby(['family_size'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


test_df['family_size'] = test_df['SibSp'] + test_df['Parch'] + 1


# In[ ]:


test_df['family_size'] = train_df['SibSp'] + train_df['Parch'] + 1


# In[ ]:


train_df.head()


# In[ ]:


train_df['Sex'].replace(['male', 'female'], [1, 0], inplace=True)
test_df['Sex'].replace(['male', 'female'], [1, 0], inplace=True)


# In[ ]:


train_df.shape


# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


train_df['Age'].fillna(round(train_df['Age'].mean()), inplace=True)
test_df['Age'].fillna(round(test_df['Age'].mean()), inplace=True)


# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


test_df['Fare'].fillna(round(test_df['Fare'].mean()), inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[ ]:


train_df.loc[train_df['Age'] <= 16, 'Age'] = 0
train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 32), 'Age'] = 1
train_df.loc[(train_df['Age'] > 32) & (train_df['Age'] <= 48), 'Age'] = 2
train_df.loc[(train_df['Age'] > 48) & (train_df['Age'] <= 64), 'Age'] = 3
train_df.loc[(train_df['Age'] > 64) & (train_df['Age'] <= 80), 'Age'] = 4


# In[ ]:


test_df.loc[test_df['Age'] <= 16, 'Age'] = 0
test_df.loc[(test_df['Age'] > 16) & (test_df['Age'] <= 32), 'Age'] = 1
test_df.loc[(test_df['Age'] > 32) & (test_df['Age'] <= 48), 'Age'] = 2
test_df.loc[(train_df['Age'] > 48) & (test_df['Age'] <= 64), 'Age'] = 3
test_df.loc[(train_df['Age'] > 64) & (test_df['Age'] <= 80), 'Age'] = 4


# In[ ]:


train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Age', ascending=True)


# In[ ]:


train_df.loc[train_df['Fare'] <= 10, 'Fare'] = 0
train_df.loc[(train_df['Fare'] > 10) & (train_df['Fare'] <= 75), 'Fare'] = 1
train_df.loc[(train_df['Fare'] > 75, 'Fare')] = 2


# In[ ]:


test_df.loc[test_df['Fare'] <= 10, 'Fare'] = 0
test_df.loc[(test_df['Fare'] > 10) & (test_df['Fare'] <= 75), 'Fare'] = 1
test_df.loc[(test_df['Fare'] > 75, 'Fare')] = 2


# In[ ]:


train_df.head()


# In[ ]:


features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'family_size']
X = train_df[features]
X_test = test_df[features]


# In[ ]:


Y = train_df['Survived']


# In[ ]:


X.head()


# In[ ]:


X_test.head()


# # Training data on different Algorithm and checking accuracy

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


# In[ ]:


models_accuracy = {}
cv = KFold(n_splits=15, random_state=13, shuffle=True)


# ****Logistic Regression****

# In[ ]:


model = LogisticRegression(solver='liblinear')
scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print(scores)
avg_score = np.round(scores.mean()*100,2)
models_accuracy['Logistic Regression'] = avg_score
print("Mean of scores = ", avg_score)


# **SVC**

# In[ ]:


model = SVC(decision_function_shape='ovr')
scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print(scores)
avg_score = np.round(scores.mean()*100,2)
models_accuracy['SVM'] = avg_score
print("Mean of scores = {}".format(np.round(scores.mean()*100, 2)))


# **KNN**

# In[ ]:


model = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print(scores)
avg_score = np.round(scores.mean()*100,2)
models_accuracy['Knn'] = avg_score
print("Mean of scores = {}".format(np.round(scores.mean()*100, 2)))


# In[ ]:


model = RandomForestClassifier(n_estimators=80)
scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print(scores)
avg_score = np.round(scores.mean()*100,2)
models_accuracy['Random Forest'] = avg_score
print("Mean of scores = {}".format(np.round(scores.mean()*100, 2)))


# In[ ]:


model = AdaBoostClassifier()
scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print(scores)
avg_score = np.round(scores.mean()*100,2)
models_accuracy['Ada Boost'] = avg_score
print("Mean of scores = {}".format(np.round(scores.mean()*100, 2)))


# In[ ]:


model = GradientBoostingClassifier(n_estimators=40)
scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print(scores)
avg_score = np.round(scores.mean()*100,2)
models_accuracy['Gradient Boost'] = avg_score
print("Mean of scores = {}".format(np.round(scores.mean()*100, 2)))


# In[ ]:


models_accuracy


# # Choosing Gradient Boosting as a Main Model

# In[ ]:


main_Model = GradientBoostingClassifier(n_estimators=40)