#!/usr/bin/env python
# coding: utf-8

# ### Lets import the necessary libraries

# In[ ]:


import math, time, random, datetime 


# In[ ]:


### Libraries for data manipulation 

import pandas as pd
import numpy as np


# In[ ]:


### Libraries for Data Visualization and to gain meaningful Data Insights

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


### I am very good at ignoring warnings ;)

import warnings
warnings.filterwarnings('ignore')


# ### Lets start with these libraries for EDA and later we will import libraries required for model building :)

# In[ ]:


#### Lets import the train & test data to check all the features/columns 

train = pd.read_csv('data/input/titanic/train.csv')
test = pd.read_csv('data/input/titanic/test.csv')

#### Also, lets have a look into the submission format for the competition

gender_submission = pd.read_csv('data/input/titanic/gender_submission.csv')


# In[ ]:


train.head()  # .head() function allow us to view the top 5 records of the dataset


# In[ ]:


test.head()


# #### One important thing to notice from 'Train' & 'Test' dataset is that 'Survived' is the dependent feature that we need to predict and the rest of the features are independent.

# In[ ]:


### Lets check the shape of the dataset 

print(train.shape) # .shape is an attribute / property of the dataset not a function.
print(test.shape) 

### Output will be in the format of (rows,columns) / "(records,features)[In a so called sophisticated way]"


# In[ ]:


### let us have a view on the submission format of the competition.

gender_submission.head()


# #### This means we need to format our predictions with respect to 'PassengerId'(unique to every person boarded the ship) and 'Survived' columns(which is the dependent feature)

# ## Lets explore the data for Data Insights or in simple terms Exploratory Data Analyis (E.D.A.)
# 
# ### We will use visualization libraries to gain meaningful insights

# In[ ]:


#### So, the first thing you should always do is try to look for the amount of null values present in your dataset.

train.isnull().sum()

### .isnull() is used to know the null values in each column in the dataset
### .sum() is used for the summation of all the null values in each column of the dataset


# In[ ]:


# Lets plot the above data in a visualization form using visualization libraries that we have earlier imported.

sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap ='viridis') 

### 'sns' is the alias that we have used for the seaborn library( you can literally use your name for alias for any libraries you import, but lets go with the standard procedure) and heatmap is an inbuilt function of the library for Data Visualization purpose 

#### you can use this link 'https://seaborn.pydata.org/' to know more about the library.


# ### One more important thing that I have forget to mention is that go through the kaggle course if you are a beginner.
# 
# ### Also Just sign in 'simplilearn.com' for "Data Science with Python course" its free for first 90 days, Its really good.
# 
# ### If you like the content in the notebook do UPVOTE, as I am also a beginner and is making a Career Transition.
# 
# ### IF YOU WANT TO LEARN TOGETHER, YOU CAN ALSO CONNECT ME ON LINKEDIN, YOU CAN FIND MY ID IN MY KAGGLE PROFILE.

# In[ ]:


# Lets check how many people survived the Titanic Disaster 

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)

print(train.Survived.value_counts()) #.value_counts() is used to count the records in features/columns


# #### We can clearly see that 342 people survived the disaster and 549 people not able to survived.

# In[ ]:


# Lets check the Pclass (Passenger Class) of the people who boarded the ship

sns.set_style('whitegrid')
sns.countplot(x='Pclass',data=train)

print(train.Pclass.value_counts())


# #### We can cleary say that max. no. of passengers are from Pclass-3, followed by Pclass-1 and Pclass-2

# In[ ]:


# Lets check the Survived feature with respect to the Pclass to find any realtion between the two

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# #### we can come to the conclusion that 'Among the Survived people, max. are from Pclass-1' and 'Among the people who had not survived, max. are from Pclass-3'

# In[ ]:


# Now let us have a look in the number of male and female who boarded the ship.

sns.set_style('darkgrid')
sns.countplot(x='Sex',data=train)

print(train.Sex.value_counts())


# #### We can say that max. number of passengers are male

# In[ ]:


# Lets check the Survived feature with respect to the Sex feature to find any realtion between the two

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='rainbow')


# #### We can clearly conclude that among people who survived the disaster, max. were female gender

# In[ ]:


# Lets check the Age feature 
pd.set_option('display.max_rows',88) # pandas function to display all rows
train.Age.value_counts()


# In[ ]:


# Also lets have a look on the null values present
train.Age.isnull().sum()


# #### So, there are 177 records where the Age data is missing

# In[ ]:


# Lets check the distribution of the Age Data by plotting a histogram 

sns.set_style('darkgrid')
sns.histplot(train['Age'].dropna(),bins=40,color='blue',kde=True)


# In[ ]:


# You can also do the same with the matplotlib library

train['Age'].hist(bins=40,color='darkred',alpha=0.85)


# #### we can cleary see that among the Age data available to us, max. passengers who boarded the ship is between (20-40) years age bracket.

# In[ ]:


# Lets know our data in dataset in a statistical way and terms

train.describe()


# In[ ]:


# Also lets have a look in their datatypes that will help us later in data cleaning/feature engineering

train.info()


# In[ ]:


# Lets check if there is any relation of Pclass with respect to Age feature so that we can derive a condition to replace the missing values in the Age data records
sns.set_style('whitegrid')
sns.boxplot(x='Pclass',y='Age',data=train,palette='rainbow')


# In[ ]:


# Lets check the 'Sibsp' feature i.e (Siblings+spouse)

sns.set_style('whitegrid')
sns.countplot(y='SibSp',data=train)

print(train.SibSp.value_counts())


# #### we can say that max. no. of passengers didn't have a siblings and spouse

# In[ ]:


# Lets have a look on Parch (The number of parents/children the passenger boarded the ship) feature

sns.set_style('whitegrid')
sns.countplot(x='Parch',data=train,palette='rainbow')

print(train.Parch.value_counts())


# #### We can say that max. no. of passengers didn't have parents/children aboarded with them.

# In[ ]:


#  lets check the ticket column to gain insights

train.Ticket.value_counts()


# In[ ]:


train.Ticket.unique()


# #### List of all the unique tickets

# In[ ]:


# Lets have a look in the Fare feature

print(len(train.Fare.unique()))
print(train.Fare.isnull().sum())


# #### There are 248 unique fare recorded in the dataset 

# In[ ]:


# Lets now check the Cabin Feature for the datset

train.Cabin.isnull().sum()


# #### So, there are 687 missing records in the Cabin feature, so during model building we will drop the column since max. values are missing and there is no way we can impute or find relation or know about these missing values

# In[ ]:


# Lets check the last column in our dataset i.e. Embarked (meaning from which place passengers boarded the ship)

print(train.Embarked.value_counts())

sns.countplot(y='Embarked', data=train)


# In[ ]:


train['Age'] = train['Age'].fillna(train['Age'].median())


# In[ ]:


sns.heatmap(train.isnull(), yticklabels= False, cbar= False, cmap= 'coolwarm')


# In[ ]:


train = train.drop('Cabin', axis=1)


# In[ ]:


train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


# If you want you can drop the 2 records from the Embarked column, but instead of dropping I will replace it with most frequent occuring category.

train['Embarked']= train['Embarked'].fillna(train['Embarked'].value_counts().index[0])


# In[ ]:


train.isnull().sum()


# ### So, we are done with our train data, now we will split our data and build our ML model to train the dataset

# In[ ]:


features= [ 'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'] # I have picked the independent features inside a list

##Let's devide the dataset

X = train[features]
y = train['Survived']


# In[ ]:


X.isnull().sum() ### Double check to make sure there is no null values in your training dataset


# ### We have cleaned our dataset, now we will use Data Preprocessing techniques or Feature Engineering to deal with Categorical Variables.

# In[ ]:


# Now let's enocde categorical values (Feature Engineering Techniques) 

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
X['Sex'] = LE.fit_transform(X['Sex'])
X['Embarked'] = LE.fit_transform(X['Embarked'])


# ## Now, we will split the dataset into training and validation format (For better understanding of what I'm saying go through the "Intro to Machine Learning" course from kaggle, its free and very good to get the foundation right :)

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state =0)


# ### Fitting a ML model, to be more specific I will be using XGBoost

# In[ ]:

