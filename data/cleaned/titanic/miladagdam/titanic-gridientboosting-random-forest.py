#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#essential library
import numpy as np
import pandas as pd 

# for eda & visualizations
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
sns.set(style='white', context='notebook', palette='deep')

#algoritms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
#evaluation

from sklearn.metrics import accuracy_score




import warnings
warnings.filterwarnings('ignore')


# # import data

# In[ ]:


train_df = pd.read_csv('data/input/titanic/train.csv')
test_df = pd.read_csv('data/input/titanic/test.csv')
PAS = list(test_df.PassengerId)


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe().T


# In[ ]:


train_df.head(10)


# In[ ]:


train_df.info()
print('_'*40)
test_df.info()


# # EDA

# In[ ]:


report = ProfileReport(test_df)
report


# In[ ]:


sns.pairplot(train_df,hue='Survived',palette='Paired');


# In[ ]:


sns.heatmap(train_df.corr(),annot=True, cmap="YlGnBu", linewidths=1)


# In[ ]:


# some information about SEX
sns.countplot(x = train_df['Sex'],hue=train_df['Survived'])


# In[ ]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index= False).mean().sort_values(by='Survived', ascending = False)


# In[ ]:


sns.countplot(x= train_df['Parch'],hue=train_df['Survived'])


# In[ ]:


sns.countplot(x= train_df['SibSp'],hue=train_df['Survived'])


# In[ ]:


# Explore Age distibution 
g = sns.kdeplot(train_df["Age"][(train_df["Survived"] == 0) & (train_df["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train_df["Age"][(train_df["Survived"] == 1) & (train_df["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# In[ ]:


train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Embarked',data=train_df,palette='rainbow')


# # Preparing Data to Create Model

# In[ ]:


train_df['Cabin'] = train_df['Cabin'].apply(lambda i: i[0] if pd.notnull(i) else 'Z')
test_df['Cabin'] = test_df['Cabin'].apply(lambda i: i[0] if pd.notnull(i) else 'Z')


# In[ ]:


print(train_df['Cabin'].value_counts(),"\n--------\n",test_df['Cabin'].value_counts())


# In[ ]:


train_df.loc[339, 'Cabin'] = 'A'
train_df['Cabin'].unique()


# In[ ]:


train_df[train_df['Cabin']=='T'].index.values


# In[ ]:


train_df['Cabin'] = train_df['Cabin'].replace(['A', 'B', 'C'], 'ABC')
train_df['Cabin'] = train_df['Cabin'].replace(['D', 'E'], 'DE')
train_df['Cabin'] = train_df['Cabin'].replace(['F', 'G'], 'FG')

test_df['Cabin'] = test_df['Cabin'].replace(['A', 'B', 'C'], 'ABC')
test_df['Cabin'] = test_df['Cabin'].replace(['D', 'E'], 'DE')
test_df['Cabin'] = test_df['Cabin'].replace(['F', 'G'], 'FG')
train_df["Cabin"].unique()


# In[ ]:


train_df.head()


# In[ ]:


train_df.drop(["Ticket", "Name", "PassengerId"], axis=1, inplace=True)
test_df.drop(["Ticket", "Name",'PassengerId'], axis=1, inplace=True)

train_df["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
test_df["Age"].fillna(test_df["Age"].median(skipna=True), inplace=True)


test_df["Fare"].fillna(test_df["Fare"].median(skipna=True), inplace=True)

train_df["Embarked"].fillna('S', inplace=True) #mode
test_df["Embarked"].fillna('S', inplace=True)


# In[ ]:


train_df.groupby("Embarked").mean()


# In[ ]:


train_df.groupby("Cabin").mean()


# In[ ]:


gender = {'male': 0, 'female': 1}
train_df.Sex = [gender[item] for item in train_df.Sex] 
test_df.Sex = [gender[item] for item in test_df.Sex] 

embarked = {'S': 0, 'Q':1, 'C': 2}
train_df.Embarked = [embarked[item] for item in train_df.Embarked] 
test_df.Embarked = [embarked[item] for item in test_df.Embarked] 

Cabins = {'Z': 0, 'FG':1, "ABC":2, 'DE': 3}
train_df.Cabin = [Cabins[item] for item in train_df.Cabin] 
test_df.Cabin = [Cabins[item] for item in test_df.Cabin] 


# In[ ]:


mask = np.triu(np.ones_like(train_df.corr(), dtype=bool))
fig, ax = plt.subplots(figsize=(16,10),dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(train_df.corr(), mask=mask, cmap="YlGnBu", vmax=.3, center=0,annot = True,
            square=True)


# In[ ]:


expected_values = train_df["Survived"]
train_df.drop("Survived", axis=1, inplace=True)
train_df.drop("Cabin", axis=1, inplace=True)
test_df.drop("Cabin", axis=1, inplace=True)


# # Scaling Data with Minmax scaler

# In[ ]:


minmax = MinMaxScaler()