#!/usr/bin/env python
# coding: utf-8

# # Business Understanding / Problem Definition

# **Titanic Survival Prediction:**
# 
# Use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

# **Variables and Their Types:**
# 
# Survival: Survival -> 0 = No, 1 = Yes
# 
# Pclass: Ticket class -> 1 = 1st, 2 = 2nd, 3 = 3rd
# 
# Sex: Sex
# 
# Age: Age in years
# 
# SibSp: # of siblings / spouses aboard the Titanic
# 
# Parch: # of parents / children aboard the Titanic
# 
# Ticket: Ticket number
# 
# Fare: Passenger fare
# 
# Cabin: Cabin number
# 
# Embarked: Port of Embarkation -> C = Cherbourg, Q = Queenstown, S = Southampton

# **Variable Notes:**
# 
# Pclass: A proxy for socio-economic status (SES)
# - 1st = Upper
# - 2nd = Middle
# - 3rd = Lower
# 
# Age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# SibSp: The dataset defines family relations in this way...
# - Sibling = brother, sister, stepbrother, stepsister
# - Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# Parch: The dataset defines family relations in this way...
# - Parent = mother, father
# - Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# # Data Understanding (Exploratory Data Analysis)

# ## Importing Librarires

# In[ ]:


# data analysis libraries:
import numpy as np
import pandas as pd

# data visualization libraries:
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')

# to display all columns:
pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split, GridSearchCV


# ## Loading Data

# In[ ]:


# Read train and test data with pd.read_csv():
train_data = pd.read_csv("data/input/titanic/train.csv")
test_data = pd.read_csv("data/input/titanic/test.csv")


# In[ ]:


# copy data in order to avoid any change in the original:
train = train_data.copy()
test = test_data.copy()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# ## Analysis and Visualization of Numeric and Categorical Variables

# ### Basic summary statistics about the numerical data

# In[ ]:


train.describe().T


# ### Classes of some categorical variables

# In[ ]:


train['Pclass'].value_counts()


# In[ ]:


train['Sex'].value_counts()


# In[ ]:


train['SibSp'].value_counts()


# In[ ]:


train['Parch'].value_counts()


# In[ ]:


train['Ticket'].value_counts()


# In[ ]:


train['Cabin'].value_counts()


# In[ ]:


train['Embarked'].value_counts()


# ### Visualization

# In general, barplot is used for categorical variables while histogram, density and boxplot are used for numerical data.

# #### Pclass vs survived:

# In[ ]:


sns.barplot(x = 'Pclass', y = 'Survived', data = train);


# #### SibSp vs survived:

# In[ ]:


sns.barplot(x = 'SibSp', y = 'Survived', data = train);


# #### Parch vs survived:

# In[ ]:


sns.barplot(x = 'Parch', y = 'Survived', data = train);


# #### Sex vs survived:

# In[ ]:


sns.barplot(x = 'Sex', y = 'Survived', data = train);


# # Data Preparation

# ## Deleting Unnecessary Variables

# In[ ]:


train.head()


# ### Ticket

# In[ ]:


# We can drop the Ticket feature since it is unlikely to have useful information
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)

train.head()


# ## Outlier Treatment

# In[ ]:


train.describe().T


# In[ ]:


# It looks like there is a problem in Fare max data. Visualize with boxplot.
sns.boxplot(x = train['Fare']);


# In[ ]:


Q1 = train['Fare'].quantile(0.25)
Q3 = train['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1- 1.5*IQR
lower_limit

upper_limit = Q3 + 1.5*IQR
upper_limit


# In[ ]:


# observations with Fare data higher than the upper limit:

train['Fare'] > (upper_limit)


# In[ ]:


train.sort_values("Fare", ascending=False).head()


# In[ ]:


# In boxplot, there are too many data higher than upper limit; we can not change all. Just repress the highest value -512- 
train['Fare'] = train['Fare'].replace(512.3292, 300)


# In[ ]:


train.sort_values("Fare", ascending=False).head()


# In[ ]:


test.sort_values("Fare", ascending=False)


# In[ ]:


test['Fare'] = test['Fare'].replace(512.3292, 300)


# In[ ]:


test.sort_values("Fare", ascending=False)


# ## Missing Value Treatment

# In[ ]:


train.isnull().sum()


# ### Age

# In[ ]:


train["Age"] = train["Age"].fillna(train["Age"].mean())


# In[ ]:


test["Age"] = test["Age"].fillna(test["Age"].mean())


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# ### Embarked

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train["Embarked"].value_counts()


# In[ ]:


# Fill NA with the most frequent value:
train["Embarked"] = train["Embarked"].fillna("S")


# In[ ]:


test["Embarked"] = test["Embarked"].fillna("S")


# ### Fare

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


test[test["Fare"].isnull()]


# In[ ]:


test[["Pclass","Fare"]].groupby("Pclass").mean()


# In[ ]:


test["Fare"] = test["Fare"].fillna(12)


# In[ ]:


test["Fare"].isnull().sum()


# ### Cabin

# In[ ]:


# Create CabinBool variable which states if someone has a Cabin data or not:

train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)

train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# ## Variable Transformation

# ### Embarked

# In[ ]:


# Map each Embarked value to a numerical value:

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)


# In[ ]:


train.head()


# ### Sex

# In[ ]:


# Convert Sex values into 1-0:

from sklearn import preprocessing

lbe = preprocessing.LabelEncoder()
train["Sex"] = lbe.fit_transform(train["Sex"])
test["Sex"] = lbe.fit_transform(test["Sex"])


# In[ ]:


train.head()


# ### Name - Title

# In[ ]:


train["Title"] = train["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
test["Title"] = test["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train.head()


# In[ ]:


train['Title'] = train['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
train['Title'] = train['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')


# In[ ]:


test['Title'] = test['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
test['Title'] = test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train[["Title","PassengerId"]].groupby("Title").count()


# In[ ]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


# Map each of the title groups to a numerical value

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5}

train['Title'] = train['Title'].map(title_mapping)


# In[ ]:


train.isnull().sum()


# In[ ]:


test['Title'] = test['Title'].map(title_mapping)


# In[ ]:


test.head()


# In[ ]:


train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# In[ ]:


train.head()


# ### AgeGroup

# In[ ]:


bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = mylabels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = mylabels)


# In[ ]:


# Map each Age value to a numerical value:
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)


# In[ ]:


train.head()


# In[ ]:


#dropping the Age feature for now, might change:
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)


# In[ ]:


train.head()


# ### Fare

# In[ ]:


# Map Fare values into groups of numerical values:
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])


# In[ ]:


# Drop Fare values:
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


# In[ ]:


train.head()


# ## Feature Engineering

# ### Family Size

# In[ ]:


train.head()


# In[ ]:


train["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1


# In[ ]:


test["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1


# In[ ]:


# Create new feature of family size:

train['Single'] = train['FamilySize'].map(lambda s: 1 if s == 1 else 0)
train['SmallFam'] = train['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
train['MedFam'] = train['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
train['LargeFam'] = train['FamilySize'].map(lambda s: 1 if s >= 5 else 0)


# In[ ]:


train.head()


# In[ ]:


# Create new feature of family size:

test['Single'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)
test['SmallFam'] = test['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
test['MedFam'] = test['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
test['LargeFam'] = test['FamilySize'].map(lambda s: 1 if s >= 5 else 0)


# In[ ]:


test.head()


# ### Embarked & Title

# In[ ]:


# Convert Title and Embarked into dummy variables:

train = pd.get_dummies(train, columns = ["Title"])
train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")


# In[ ]:


train.head()


# In[ ]:


test = pd.get_dummies(test, columns = ["Title"])
test = pd.get_dummies(test, columns = ["Embarked"], prefix="Em")


# In[ ]:


test.head()


# ### Pclass

# In[ ]:


# Create categorical values for Pclass:
train["Pclass"] = train["Pclass"].astype("category")
train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc")


# In[ ]:


test["Pclass"] = test["Pclass"].astype("category")
test = pd.get_dummies(test, columns = ["Pclass"],prefix="Pc")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:





# # Modeling, Evaluation and Model Tuning

# ## Spliting the train data

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20, random_state = 0)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()