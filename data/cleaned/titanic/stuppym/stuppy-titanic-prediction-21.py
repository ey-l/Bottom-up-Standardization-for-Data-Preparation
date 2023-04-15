#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Prediction by the Southern Stuppy Family
# *Import necessary libraries before reading cvs files*

# In[ ]:



import seaborn


# In[ ]:


# Import libraries 
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from collections import Counter


import missingno 

# Machine learning models 
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier

# Model evaluation
from sklearn.model_selection import cross_val_score

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Remove warnings
import warnings 
warnings.filterwarnings('ignore')


# In[ ]:


# Read cvs files - mainly train and the test
train = pd.read_csv('data/input/titanic/train.csv')
test = pd.read_csv('data/input/titanic/test.csv')
gs = pd.read_csv('data/input/titanic/gender_submission.csv')


# ## Checking Train and Test.csv

# In[ ]:


train.head()


# In[ ]:


test.head() # this doesn't have 'Survived' while train has 


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


gs.head() #This prediction assumes only female passengers survival #.


# # Finding Missing information for train and test.csv

# In[ ]:


# non-null counts for train.csv
train.info()
### Age, Cabin and Embarked columns are missing information - while others have all 891 non-null 


# In[ ]:


test.info() #also 'Age', 'Cabin', and 'Fare'(only one) are missing in test.csv


# 
# # Find out missing information - NA information

# In[ ]:


# Missing data in training set by columns

train.isnull().sum().sort_values(ascending=False)
# Cabin and Ages are missing many data, while Embarked is missing 2 items. 


# In[ ]:


test.isnull().sum().sort_values(ascending=False)
# as well as the test data,,,cabin and age have many missing data


# In[ ]:


# Showing missing data by using `missingno.matrix` in train set
missingno.matrix(train)


# In[ ]:


# Missing data in test set
missingno.matrix(test)


# In[ ]:


# Summary statistics for training set
train.describe()


# In[ ]:


test.describe() #For Test set


# ### Train.csv is missing below information. 
# - Cabin 687
# - Age   177
# - Embarked  2
# ### Test.csv is missing below information.
# - Cabin 327
# - Age    86 
# - Fare    1
# 
# We don't want to drop them, but fill information to uterlize...

# ## Focusing on missing `Embarked` column for train.csv

# In[ ]:


# Find out missing `Embarked -2 items`
train[['Embarked']] = train[['Embarked']].fillna('Unknown')
train[train['Embarked'] == 'Unknown']
# So both of these ladies stayed at the same Cabin. They both survived. 
# Where they could Embarked? It is likely we can find the answer by looking up the Cabin? 


# ## Focusing on missing `Age` column for train.csv

# In[ ]:


# Fill out missing `Age` as 99 years 
train[['Age']] = train[['Age']].fillna(99)
train[train['Age'] == 99]#Show the missing age ones as 99 years old for now. 


# ## Focusing on missing `Cabin` column for train.csv

# In[ ]:


# Fill NaN in Cabin Column
train[['Cabin']] = train[['Cabin']].fillna('Unknown')
train[train['Cabin'] == 'Unknown']


# ## Move on to `Test.csv` to fill in missing information

# ## Focusing on missing `Cabin` column for test.csv

# In[ ]:


# Fill NaN in Cabin Column
test[['Cabin']] = test[['Cabin']].fillna('Unknown')
test[test['Cabin'] == 'Unknown']


# ## Focusing on missing `Age` column for test.csv

# In[ ]:


# Fill out missing `Age` as 99 years 
test[['Age']] = test[['Age']].fillna(99)
test[test['Age'] == 99]#Show the missing age ones as 99 years old for now. 


# ## Focusing on missing `Fare` column for test.csv - only one but just in case...

# In[ ]:


nan_fare = test[pd.isnull(test).any(axis=1)]
test[['Fare']] = test[['Fare']].fillna(35) # pick the mean of the test Fare $
test[test['Fare'] == 35]


# ### Check to make sure we filled out all the missing information for both Train and Test.csv

# In[ ]:


#train.csv
train[pd.isnull(train).any(axis=1)]


# In[ ]:


#test.csv
test[pd.isnull(test).any(axis=1)]


# ## Good. So we filled out everything in the columns and there aren't any empty spots. 
# 
# ## Now, clean up the columns and rows for both Train and Test.csv. This will help to read these documents later! 

# In[ ]:


#Find out first 2 letters of Cabin information in train. If it says 'Un', it means it is unknown
train['Cabin'] = train['Cabin'].str[0:2]
train['Cabin']


# In[ ]:


#Find out first 2 letters of Cabin information in test. If it says 'Un', it means it is unknown
test['Cabin'] = test['Cabin'].str[0:2]
test['Cabin']


# In[ ]:


# Combine Cabin, Pclass and Sex and get the survived average number by these three columns 
cabin_survived = pd.DataFrame(train.groupby(['Cabin', 'Pclass','Sex']).agg({'Survived': 'mean'}, inplace=True, index=False))
cabin_survived


# In[ ]:


cabin_survived.columns # What type of column cabin_survived? 


# In[ ]:


# Added the above `cabin_survived` columns to the train to show how cabin # affected to the survival #s
train=train.merge(cabin_survived, on=['Cabin', 'Pclass', 'Sex'], how='left')
train #It shows as "Survived_y"


# In[ ]:


# Rename 'Survived_x' and 'Survived_y' to 'Survived' and 'Cabin IND(Indicator)' -based on ticket information 
train.rename(columns={'Survived_x': 'Survived', 'Survived_y': 'Cabin IND'}, inplace=True)
train.head()


# ## Move on to test.csv to do the same as the train.csv

# In[ ]:


# Take a look at test.csv
test.head()


# In[ ]:


#Add Cabin IND to test data also - since it doesn't include 'Survived' originally
test=test.merge(cabin_survived, on=['Cabin', 'Pclass', 'Sex'], how='left')
test.head()


# In[ ]:


#Rename 'Survived' to 'Cabin IND'-based on ticket information 
test.rename(columns={'Survived': 'Cabin IND'}, inplace=True)
test.head()


# In[ ]:


# Drop the douplicated column - Survived (Cabin IND) from the test  & Cabin as well since we don't need it anymore 
test.drop(['Cabin'], axis=1, inplace=True)
test.head()


# ## By filling missing na as 99, you can move on to the next question. 
# - Do they have parents or child(ren) ? Any siblings? 
# - How much did they pay for the fare? 
# - Which class were they in ? 
# 
# ## Let's go back and forth on train and test.csv to see you can find out age, ticket, and class situations. 

# In[ ]:


nines= train[train['Age']==99]
nines


# In[ ]:


# To see if class and embarked are related- if you got on the certain embarked area, that means you are rich or poor?
# Find out there two ladies - where they embarked from?
classes = train.groupby(['Ticket']).agg({'Survived':['mean','min','max'],'Age':'nunique'}, inplace=True, index=True)
classes.head()


# In[ ]:


train['Ticket_kind'] = train['Ticket'].str[0:1]
train['Ticket_kind']


# In[ ]:


test['Ticket_kind'] = test['Ticket'].str[0:1]
test['Ticket_kind']


# In[ ]:


# Find out if ticket class affected on the passenger survival or not.
ticket_survived = pd.DataFrame(train.groupby(['Ticket_kind','Pclass']).agg({'Ticket': 'nunique','Survived': 'mean'}, inplace=True, index=False))
ticket_survived


# ### It does! If you have the smaller ticket # and if the Pclass is 1, you would likely be survived compared to bigger ticket numbers. 

# In[ ]:


# Added the above `ticket_survived` columns to the train to show how ticket # affected to the survival #s
train=train.merge(ticket_survived, on=['Ticket_kind', 'Pclass'], how='left')
train


# In[ ]:


# Added the above `ticket_survived` columns to the test to show how ticket # affected to the survival #s
test=test.merge(ticket_survived, on=['Ticket_kind', 'Pclass'], how='left')
test


# In[ ]:


train.columns #See the total column names so far. 


# In[ ]:


train.drop(['Ticket_x','Ticket_y','Ticket_kind'],axis=1, inplace=True)
train # Now get rid of repeated columns that added to the original csv.


# In[ ]:


# Rename 'Survived' -based on ticket information 
train.rename(columns={'Survived_x': 'Survived', 'Survived_y': 'Ticket IND'}, inplace=True)
train.head()


# In[ ]:


# Drop Cabin column since we created Cabin_IND
train.drop(['Cabin'], axis=1, inplace=True)
train.head()


# ## Move on to test.csv to do the same as above. 

# In[ ]:


test.head()


# In[ ]:


test.drop(['Ticket_kind', 'Ticket_x', 'Ticket_y'], axis=1, inplace=True)


# In[ ]:


test['Ticket IND'] = test['Survived'] # rename 'Survived' to 'Ticket IND'


# In[ ]:


test.head()


# In[ ]:


test.drop('Survived', axis=1, inplace=True)


# In[ ]:


test.head()


# # Let's go back to the train.csv to focus on `Age` column. What types of age groups survived? Boys or Girls? 

# In[ ]:


#find out age range who are younger than 19
# Make a chart to see the survival rate
children = train[train['Age'] < 19]
children


# In[ ]:


sns.countplot(data=children, x='Pclass', hue='Survived')
# About a half of children on the passenger class 3 didn't survived. 


# In[ ]:


sns.countplot(data=children,x='Sex', hue='Survived')
# girls survived compared to boys 


# # Let's only focus on `Sex` category- not numerical values

# In[ ]:


# Value counts of the sex column

train['Sex'].value_counts(dropna=False) #dropna - groups all the outcome together


# In[ ]:


# Mean of survival by sex 

train[['Sex','Survived']].groupby('Sex', as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


sns.barplot(x='Sex', y ='Survived', data=train)
plt.ylabel('Survival Probability')
plt.title('Survival Probablity by Gender')


# ### Wow! You rather not to get in the Titanic if you are male!!

# ## Let's focus on `Pclass` only.  

# In[ ]:


# Value counts of the Pcalss column

train['Pclass'].value_counts(dropna=False)


# In[ ]:


# Mean of survival by passenger class

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


test['Pclass'].value_counts(dropna=False)


# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=train)
plt.ylabel('Survival Probability')
plt.title('Survival Probability by Passenger Class')


# ### Passenger Class 1 survived all the way! 
# 
# ## Let's combine gender and passenger together to see if there are any differences

# In[ ]:


# Survival by genger and passenger class 

g= sns.factorplot(x='Pclass', y='Survived', hue='Sex', data=train, kind = 'bar')
g.despine(left=True)
plt.ylabel('Survival Probability')
plt.title('Survival Probability by Passenger Class')


# ## If you got in Passenger 1 and female, you were likely to be survived! 

# ## Let's see if the port is something to do with survival rates. Focus on- `Embarked`

# In[ ]:


# Value counts of the Embarked column

train['Embarked'].value_counts(dropna=False)


# In[ ]:


# Mean of survival by point of embarkation

train[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Seems Unknown could be incluided to 'C' - Embarked


# In[ ]:


sns.barplot(x='Embarked', y='Survived', data=train)
plt.ylabel('Survival Probability')
plt.title('Survival Probability by Point of Embarkation')


# ## If you embarked at C port, then you survived the most....
# 
# ## Clean up `Name` by getting rid of their actual names, and pull their `Title` from it. Add the column as `Title`

# In[ ]:


# Get title from names

train['Title'] = [name.split(',')[1].split('.')[0].strip() for name in train['Name']]
train[['Name', 'Title']].head()
train.head()


# In[ ]:


# since we now have 'Title' column in the train table, get rid of Name 
train.drop('Name', axis=1, inplace=True)
train.head()


# In[ ]:


train['Title'].value_counts()
# See how many different titles are


# In[ ]:


# Combine titles that aren't more than 3 or less
train['Title'] = train['Title'].replace(['Dr', 'Rev', 'Col','Major','Lady','Jonkheer','Don','Capt','the Countess','Sir','Dona'], 'Others')
train['Title'] = train['Title'].replace(['Mlle', 'Mme', 'Ms'], 'Miss')
train.head()


# In[ ]:


sns.countplot(data=train, x = 'Title')


# In[ ]:


title_survived = train[['Title', 'Survived']].groupby(['Title'], as_index = False).mean().sort_values(by='Survived', ascending=False)
title_survived

#train[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train=train.merge(title_survived, on='Title',how='left')
train


# In[ ]:


train['Survived'] = train['Survived_x'] # Change the column names to Survived and Title Indicator.
train['Title IND'] = train['Survived_y']


# In[ ]:


train.head()


# In[ ]:


train.drop(['Survived_x','Title','Survived_y'], axis=1, inplace=True) 
# We can get rid of 'Survived_x', 'Title', and 'Survived_y' now.


# In[ ]:


train.head() # train.csv is so much cleaner! 


# ### Now, train.csv is so much cleaner! Although, but the below are other items we need to do...

# * Identify age ranges and then concatenate sex and age range 
# * Find mean survival of each combination
# * Drop the original age and sex 
# 
# # Let's do the same for test.csv to clean up for the document.

# In[ ]:


# Get title from names for test

test['Title'] = [name.split(',')[1].split('.')[0].strip() for name in test['Name']]
test[['Name', 'Title']].head()
test.head()


# In[ ]:


# since we now have 'Title' column in the train table, get rid of Name 
test.drop('Name', axis=1, inplace=True)
test.head()


# In[ ]:


test['Title'].value_counts()
# See how many different titles are


# In[ ]:


# Combine titles that aren't more than 3 or less
test['Title'] = test['Title'].replace(['Dr', 'Rev', 'Col','Major','Lady','Jonkheer','Don','Capt','the Countess','Sir','Dona'], 'Others')
test['Title'] = test['Title'].replace(['Mlle', 'Mme', 'Ms'], 'Miss')
test.head()


# In[ ]:


test=test.merge(title_survived, on=['Title'], how='left')
test


# In[ ]:


# Drop title and rename survived to Title IND
test.drop('Title', axis=1, inplace=True)
test.head()


# In[ ]:


test['Title IND'] = test['Survived']
test.head()


# In[ ]:


test.drop('Survived', axis=1, inplace=True)


# In[ ]:


test.head()


# ### Focusing on Age now 
# Fill missing information for Test Age column -  83 are missing

# In[ ]:


train.head()


# In[ ]:


test.loc[test['Age'] <= 16.0, 'Age'] = 1
test.loc[(test['Age'] > 16.0) & (test['Age'] <= 35.0), 'Age'] = 2
test.loc[(test['Age'] >35.0) & (test['Age'] <= 50.0), 'Age']= 3
test.loc[(test['Age'] > 50.0) & (test['Age'] <=70.0), 'Age']= 4
test.loc[(test['Age'] > 70.0) & (test['Age'] <= 98.0), 'Age']= 5
test.loc[test['Age'] == 99.0, 'Age'] = 6


# * Age 1 = 16 and younger
# * Age 2 = 17 and 35 years old 
# * Age 3 = 36 and 50 years old
# * Age 4 = 51 and 70 years old
# * Age 5 = 71 and 98 years old
# * Age 6 = missing information, we put them as 99 years old 

# In[ ]:


sns.countplot(data=train, x='Age')


# In[ ]:


# Group by sex. age. and survived 
age_sex = train[['Sex','Age', 'Survived']].groupby(['Sex', 'Age'], as_index = False).mean().sort_values(by ='Survived', ascending=False)
age_sex


# * Seems like older ladies survived more than any other aged folks- males had no chance compared to women.... 

# In[ ]:


# Added the above `age_sex` columns to the train to show how Sex and Age affected to the survival #s
train=train.merge(age_sex, on=['Sex','Age'], how='left')
train


# In[ ]:


test=test.merge(age_sex, on=['Sex','Age'], how='left')
test


# In[ ]:


# Rename Survvied_y from sex_age combination

train['Sex Age IND'] = train['Survived_y'] 
train['Survived'] = train['Survived_x']

train.head()


# In[ ]:


# Drop columns that we no longer needed
train.drop(['Survived_y', 'Sex', 'Age', 'Survived_x'], axis=1, inplace=True)
train.head()


# In[ ]:


# Would it be any relations for survived #s for Embarked? Let's focus on the column
sns.countplot(x='Embarked', data=train, hue='Survived')


# * So if you got in S port, that means you died more than any other ports. (600 total - 400 died from S port)
# * What about Pclass relation of Embarked? Let's put hue= plcass
# * Seems like Q port has more third class folks, although more people died from S-port.

# In[ ]:


sns.countplot(x='Embarked', data=train, hue='Pclass')


# In[ ]:


# Group by embarked, pclass. and survived 
class_port = train[['Embarked','Pclass', 'Survived']].groupby(['Embarked', 'Pclass'], as_index = False).mean().sort_values(by ='Survived', ascending=False)
class_port


# In[ ]:


# Added the above `emberked` and 'pclass' columns to the train to show how embarked and plcass affected to the survival #s
train=train.merge(class_port, on=['Embarked','Pclass'], how='left')
train


# In[ ]:


# Rename Survvied_y from Embarked and Pclass combination

train['Embarked Pclass IND'] = train['Survived_y'] 
train['Survived'] = train['Survived_x']

train.head()


# In[ ]:


# Drop columns that we no longer needed
train.drop(['Survived_y', 'Embarked', 'Survived_x'], axis=1, inplace=True)
train.head()


# In[ ]:


# Group by SibSp and Parch and survived 
Sib_Parch = train[['SibSp', 'Parch', 'Survived']].groupby(['SibSp','Parch'], as_index = False).mean().sort_values(by='Survived', ascending=False)
Sib_Parch


# In[ ]:


# Added the above `` and '' columns to the train to show how embarked and plcass affected to the survival #s
train=train.merge(Sib_Parch, on=['SibSp','Parch'], how='left')
train


# In[ ]:


train.columns


# In[ ]:


# Rename Survvied_y column to 'Family IND'

train['Family IND'] = train["Survived_y"] 
train.head()


# In[ ]:


# Change Survived_x to Survived
train['Survived'] = train['Survived_x']
train.head()


# In[ ]:


# Drop Survived_y & Survived_x
train.drop(['Survived_x', 'Survived_y'], axis=1, inplace=True)
train.head()


# In[ ]:


# Since Family IND has SibSp and Parch mean information, drop SibSp and Parch columns. 
train.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


# Make Sex Age IND for test

test['Sex Age IND'] = test['Survived'] 

test.head()


# In[ ]:


# Remove 'Sex', 'Age', and 'Survived' columns
test.drop(['Sex','Age'], inplace=True, axis=1)
test.head()


# In[ ]:


# Group by embarked, pclass. and survived 
class_port = test[['Embarked','Pclass', 'Survived']].groupby(['Embarked', 'Pclass'], as_index = False).mean().sort_values(by ='Survived', ascending=False)
class_port


# In[ ]:


# Added the above `emberked` and 'pclass' columns to the test to show how embarked and plcass affected to the survival #s
test=test.merge(class_port, on=['Embarked','Pclass'], how='left')
test


# In[ ]:


# Rename Survvied_y from Embarked and Pclass combination

test['Embarked Pclass IND'] = test['Survived_y'] 
test['Survived'] = test['Survived_x']

test.head()


# In[ ]:


# Drop columns that we no longer needed
test.drop(['Survived_y', 'Embarked', 'Survived_x'], axis=1, inplace=True)
test.head()


# In[ ]:


# Group by SibSp and Parch and survived 
Sib_Parch_Test = test[['SibSp', 'Parch', 'Survived']].groupby(['SibSp','Parch'], as_index = False).mean().sort_values(by='Survived', ascending=False)
Sib_Parch_Test


# In[ ]:


# Added the above `` and '' columns to the test to show how embarked and plcass affected to the survival #s
test=test.merge(Sib_Parch, on=['SibSp','Parch'], how='left')
test


# In[ ]:


test.columns


# In[ ]:


# Rename Survvied_y column to 'Family IND'
test['Family IND'] = test["Survived_y"] 
test.head()


# In[ ]:


# Change Survived_x to Survived
test['Survived'] = test['Survived_x']
test.head()


# In[ ]:


# Drop Survived_y & Survived_x
test.drop(['Survived_x', 'Survived_y'], axis=1, inplace=True)
test.head()


# In[ ]:


# Since Family IND has SibSp and Parch mean information, drop SibSp and Parch columns. 
test.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# ### We know `Cabin IND` and `Family IND` columns are missing some info. Let's fill out them with their average numbers. 

# In[ ]:


test[test['Cabin IND'].isna()]=test['Cabin IND'].mean()

test[test['Cabin IND'].isna()]


# In[ ]:


test[test['Family IND'].isna()]=test['Family IND'].mean()

test[test['Family IND'].isna()]


# In[ ]:


test.info()
train.info()


# In[ ]:


test.head()


# ## We need to get rid of `Survived` from the test that we added so that we can start testing!

# In[ ]:


X_train = train.drop('Survived', axis=1)
Y_train = train['Survived']
X_test = test.drop('Survived', axis=1)
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)


# In[ ]:


X_test.head()


# # Run with different testing models 
# ## Logistic Regression

# In[ ]:


logreg = LogisticRegression()