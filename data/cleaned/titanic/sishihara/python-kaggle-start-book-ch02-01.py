#!/usr/bin/env python
# coding: utf-8

# This notebook is a sample code with Japanese comments.
# 
# # 2.1 まずはsubmit！　順位表に載ってみよう

# In[ ]:


import numpy as np
import pandas as pd


# ## データの読み込み

# In[ ]:





# In[ ]:


train = pd.read_csv('data/input/titanic/train.csv')
test = pd.read_csv('data/input/titanic/test.csv')
gender_submission = pd.read_csv('data/input/titanic/gender_submission.csv')


# In[ ]:


gender_submission.head()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


data = pd.concat([train, test], sort=False)


# In[ ]:


data.head()


# In[ ]:


print(len(train), len(test), len(data))


# In[ ]:


data.isnull().sum()


# ## 特徴量エンジニアリング

# ### 1. Pclass

# ### 2. Sex

# In[ ]:


data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)


# ### 3. Embarked

# In[ ]:


data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


# ### 4. Fare

# In[ ]:


data['Fare'].fillna(np.mean(data['Fare']), inplace=True)


# ### 5. Age

# In[ ]:


age_avg = data['Age'].mean()
age_std = data['Age'].std()

data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)


# In[ ]:


delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)


# In[ ]:


train = data[:len(train)]
test = data[len(train):]


# In[ ]:


y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# ## 機械学習アルゴリズム

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


clf = LogisticRegression(penalty='l2', solver='sag', random_state=0)


# In[ ]:

