#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Prediction:
# 
# Use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.
# 

# ### Variables and Their Types:
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
# 
# #### Types
# 
# Numerical Features: Age , Fare , SibSp , Parch 
# 
#     Age: float
#     Fare: float
#     SibSp: int
#     Parch: int   
#     
# Categorical Features: Survived, Sex, Embarked, Pclass
# 
#     Survived: int
#     Sex: string
#     Embarked: string
#     Pclass: int
#     
# Alphanumeric Features: Ticket, Cabin
# 
#     Ticket: string
#     Cabin: string
# 

# ### Variable Notes:
# 
# Pclass: A proxy for socio-economic status (SES)
# 
#     1st = Upper
#     2nd = Middle
#     3rd = Lower
# 
# Age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# SibSp: The dataset defines family relations in this way...
# 
#     Sibling = brother, sister, stepbrother, stepsister
#     Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# Parch: The dataset defines family relations in this way...
# 
#     Parent = mother, father
#     Child = daughter, son, stepdaughter, stepson Some children travelled only with a nanny, therefore parch=0 for them.
# #### Now  we have an idea of what kinds of features we're working with.
# 

# ## 1) Import Necessary Libraries
# First off, we need to import several Python libraries such as numpy, pandas, matplotlib and seaborn.

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


# ## 2) Loading Data
# It's time to read in our training and testing data using pd.read_csv, and take a first look at the training data using the describe() function. We will take train and test copies.
# 

# In[ ]:


#import train and test CSV files

train_data = pd.read_csv("data/input/titanic/train.csv")
test_data = pd.read_csv("data/input/titanic/test.csv")


# In[ ]:


# copy data in order to avoid any change in the original:

train = train_data.copy()
test = test_data.copy()


# In[ ]:


#take a look at the training data

train.describe(include="all")


# ## 3) Data Analysis
# We're going to consider the features in the dataset and how complete they are.
# 

# In[ ]:


#get information about the dataset

train.info()


# In[ ]:


#get a list of the features within the dataset

print(train.columns)


# In[ ]:


#head 

train.head()


# In[ ]:


#head 

test.head()


# In[ ]:


#tail

train.tail()


# In[ ]:


#see a sample of the dataset to get an idea of the variables

train.sample(5)


# In[ ]:


#check for any other unusable values

print(pd.isnull(train).sum())


# We can see that except for the above mentioned missing values, not NaN values exist.

# In[ ]:


#see a summary of the training dataset

train.describe().T


# In[ ]:


100*train.isnull().sum()/len(train)


# Some Observations:
# 
# There are a total of 891 passengers in our training set.
# The Age feature is missing approximately 19.8% of its values. I'm guessing that the Age feature is pretty important to survival, so we should probably attempt to fill these gaps.
# The Cabin feature is missing approximately 77.1% of its values. Since so much of the feature is missing, it would be hard to fill in the missing values. We'll probably drop these values from our dataset.
# The Embarked feature is missing 0.22% of its values, which should be relatively harmless.

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


# # 4) Data Visualization
# It's time to visualize our data so we can see whether our predictions were accurate! 
# In general, barplot is used for categorical variables while histogram, density and boxplot are used for numerical data.
# 
# 

# ##### Sex Feature

# In[ ]:


#draw a bar plot of survival by sex

sns.barplot(x="Sex", y="Survived", data=train)

#print percentages of females vs. males that survive

print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)


# As predicted, females have a much higher chance of survival than males. The Sex feature is essential in our predictions.

# ##### Pclass Feature

# In[ ]:


#draw a bar plot of survival by Pclass

sns.barplot(x="Pclass", y="Survived", data=train)

#print percentage of people by Pclass that survived

print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)


# As predicted, people with higher socioeconomic class had a higher rate of survival. (62.9% vs. 47.3% vs. 24.2%)
# 

# ##### SibSp Feature

# In[ ]:


#draw a bar plot for SibSp vs. survival

sns.barplot(x="SibSp", y="Survived", data=train)

#I won't be printing individual percent values for all of these.

print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 3 who survived:", train["Survived"][train["SibSp"] == 3].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 4 who survived:", train["Survived"][train["SibSp"] == 4].value_counts(normalize = True)[1]*100)


# In general, it's clear that people with more siblings or spouses aboard were less likely to survive.
# 
# However, contrary to expectations, people with no siblings or spouses were less to likely to survive than those with one or two. (34.5% vs 53.4% vs. 46.4%)
# 

# ##### Parch Feature

# In[ ]:


#draw a bar plot for Parch vs. survival

sns.barplot(x="Parch", y="Survived", data=train)



# People with less than four parents or children aboard are more likely to survive than those with four or more.
# 
# Again, people traveling alone are less likely to survive than those with 1-3 parents or children.

# ##### Cabin Feature
# 

# I think the idea here is that people with recorded cabin numbers are of higher socioeconomic class, and thus more likely to survive. 
# 
# 

# In[ ]:



train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

#calculate percentages of CabinBool vs. survived

print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)

#draw a bar plot of CabinBool vs. survival

sns.barplot(x="CabinBool", y="Survived", data=train)




# People with a recorded Cabin number are, in fact, more likely to survive. (66.6% vs 29.9%)

# ## 5) Cleaning Data

# Time to clean our data to account for missing values and unnecessary information!
# 
# Looking at the Test Data
# 
# Let's see how our test data looks!
# 

# In[ ]:


test.describe().T


# #### Cabin Feature
# 

# In[ ]:


# Create CabinBool variable which states if someone has a Cabin data or not:

train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)

train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:





# In[ ]:


test.isnull().sum()


# In[ ]:


print(pd.isnull(train.CabinBool).sum())


# #### Ticket Feature

# In[ ]:


# We can drop the Ticket feature since it is unlikely to have useful information

train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)

train.head()


# In[ ]:


train.describe().T


# In[ ]:


# It looks like there is a problem in Fare max data. Visualize with boxplot.

sns.boxplot(x = train['Fare']);


# In[ ]:


Q1 = train['Fare'].quantile(0.05)
Q3 = train['Fare'].quantile(0.95)
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

train['Fare'] = train['Fare'].replace(512.3292, 270)


# In[ ]:


train.sort_values("Fare", ascending=False).head()


# In[ ]:


train.sort_values("Fare", ascending=False)


# In[ ]:


test.sort_values("Fare", ascending=False)


# In[ ]:


test['Fare'] = test['Fare'].replace(512.3292, 270)


# In[ ]:


test.sort_values("Fare", ascending=False)


# #### Name Feature

# We can drop the name feature now that we've extracted the titles.

# In[ ]:


#drop the name feature since it contains no more useful information.

train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# In[ ]:


train.describe().T


# ### Missing Value Treatment
# 

# In[ ]:


train.isnull().sum()


# #### Age Feature

# 
# We'll fill in the missing values in the Age feature.
# 

# In[ ]:


train["Age"] = train["Age"].fillna(train["Age"].median())


# In[ ]:


test["Age"] = test["Age"].fillna(test["Age"].median())


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train.describe().T


# #### Embarked Feature

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


#now we need to fill in the missing values in the Embarked feature

print("Number of people embarking in Southampton (S):")
southampton = train[train["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = train[train["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = train[train["Embarked"] == "Q"].shape[0]
print(queenstown)


# In[ ]:


train["Embarked"].value_counts()


# In[ ]:


#replacing the missing values in the Embarked feature with S

train = train.fillna({"Embarked": "S"})


# In[ ]:


test = test.fillna({"Embarked": "S"})


# In[ ]:


train.Embarked


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


print(pd.isnull(train.Embarked).sum())


# #### Fare Feature

# It's time separate the fare values into some logical groups as well as filling in the single missing value in the test dataset.

# In[ ]:


test[test["Fare"].isnull()]


# In[ ]:


test[["Pclass","Fare"]].groupby("Pclass").mean()


# In[ ]:


test["Fare"] = test["Fare"].fillna(12)


# In[ ]:


test["Fare"].isnull().sum()


# In[ ]:


#check train data

train.head()


# In[ ]:


#check test data

test.head()


# ### Variable Transformation
# 

# #### Sex Feature

# In[ ]:


#map each Sex value to a numerical value

sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()


# #### Embarked Feature

# In[ ]:


#map each Embarked value to a numerical value
from sklearn import preprocessing
lbe = preprocessing.LabelEncoder()
train["Embarked"] = lbe.fit_transform(train["Embarked"])
test["Embarked"] = lbe.fit_transform(test["Embarked"])
train.head()


# #### AgeGroup

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


# #### Fare

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


# ## 6) Feature Engineering
# 

# ##### Family Size

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


# #### Embarked & Title

# In[ ]:


# Convert Title and Embarked into dummy variables:

train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")


# In[ ]:


train.head()


# In[ ]:



test = pd.get_dummies(test, columns = ["Embarked"], prefix="Em")


# In[ ]:


test.head()


# ## 7) Modeling

# #### Spliting the train data
# 

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


# #### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()