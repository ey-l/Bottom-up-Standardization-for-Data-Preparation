#!/usr/bin/env python
# coding: utf-8

# <h2> Importing the required Libraries </h2>

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree,svm
from sklearn.metrics import accuracy_score


# <h2> Reading the Training Data using Pandas </h2>

# In[ ]:


train_data = pd.read_csv('data/input/titanic/train.csv')
train_data


# In[ ]:


print('The shape of our training set: %s passengers and %s features'%(train_data.shape[0],train_data.shape[1]))


# In[ ]:


train_data.info()


# <h2> Checking for Null values </h2>

# In[ ]:


train_data.isnull().sum()


# <h4> So, there are 177 null values in 'Age' Column, 687 null values in 'Cabin' Column and 2 null values in 'Embarked' Column </h4>

# <h2>  Plotting a heat map to see the correlation between the parameters and the target variable (Survived) </h2>

# In[ ]:


sns.heatmap(train_data[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr(), annot = True)
sns.set(rc={'figure.figsize':(12,10)})


# <h4> From the heatmap we can say that those who have paid higher fares have a better chance of Survival </h4>

# <h2> Plotting Bar Graph to visualize the Surviving Probability w.r.t SibSp (Siblings/Spouses)</h2>

# In[ ]:


train_data['SibSp'].unique()
sns.catplot(x = "SibSp", y = "Survived", data = train_data, kind="bar", height = 8)


# <h4> From the Bar Graph we can say that Passengers with 1 or 2 siblings have a better chance of survival than those have more than 2 siblings</h4>

# <h2> Plotting a graph so as to see the distribution of age w.r.t Target Variable(Survival) </h2>

# In[ ]:


sns.FacetGrid(train_data, col="Survived", height = 7).map(sns.distplot, "Age").set_ylabels("Survival Probability")


# <h4> From the distribution plot we can say that people with more age have a lesser chance of survival than people with less age </h4>

# <h2> Using Barplot to visulalize the survival rate w.r.t gender of survivor </h2>

# In[ ]:


sns.barplot(x="Sex", y="Survived", data=train_data)


# <h4> From above it is crystal clear that "There were more Female survivors than compared to Male survivors </h4>

# <h2> Analyzing the Pclass column to get the survival chance analysis using bar plot </h2>

# In[ ]:


sns.catplot(x = "Pclass", y="Survived", data = train_data, kind="bar", height = 6)


# <h3>Chances of survival :-</h3>
# <h3>1st Class > 2nd Class > 3rd Class</h3>

# <h2> Embarked column analysis and correction of null values </h2>

# In[ ]:


train_data['Embarked'].value_counts(), train_data['Embarked'].isnull().sum()


# <h3> So we have 2 null values in 'Embarked' Column of the data set </h3>

# <h4> Replacing the null values with the most frequent value 'S' </h4>

# In[ ]:


train_data["Embarked"] = train_data["Embarked"].fillna('S')
train_data.isnull().sum()


# In[ ]:


sns.catplot(x="Embarked", y="Survived", data=train_data, height = 5, kind="bar")


# In[ ]:


sns.catplot(x="Pclass", col="Embarked", data = train_data, kind="count", height=7)


# <h2> Passengers embarked from C station had the most of the 1st Class booking and so the Survival chances of Passengers embarked from C Station is High </h2>

# <h2> Handling missing values / null values from the 'Age' Column and replacing them with random values within the range of the Average Age - Standard Age and Average Age + Standard Age </h2>

# In[ ]:


mean_age = train_data["Age"].mean()
std_age = train_data["Age"].std()
mean_age, std_age


# In[ ]:


random_age = np.random.randint(mean_age-std_age, mean_age+std_age, size = 177)
age_slice = train_data["Age"].copy()
age_slice[np.isnan(age_slice)] = random_age
train_data["Age"] = age_slice


# In[ ]:


train_data.isnull().sum()


# <h2> Dropping Un-necessary Columns from the Dataset </h2>

# In[ ]:


list_column_to_drop = ["PassengerId", "Ticket", "Cabin", "Name"]
train_data.drop(list_column_to_drop, axis=1, inplace=True)
train_data.head(10)


# <h2> Converting Categorical Variables (Sex, Embarked) to Numeric </h2>

# In[ ]:


genders = {"male":0, "female":1}
train_data["Sex"] = train_data["Sex"].map(genders)

ports = {"S":0, "C":1, "Q":2}
train_data["Embarked"] = train_data["Embarked"].map(ports)


# In[ ]:


train_data.head()


# <h1> Building Machine Learning Model </h1>

# In[ ]:


df_train_x = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
df_train_y = train_data[['Survived']] #Target Variable 


# <h2> Train, Test and Splitting </h2>

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.20, random_state=18)


# <h2> Fitting the machine learning model on 4 different classification algorithms namely Random Forest Classifier, K-Neighbor Classifier, Decision Tree Classifier, Support Vector Machine and Logistic Regression and comparing them </h2>

# <h3> 1. Random Forest Classifier </h3>

# In[ ]:


# Creating alias for Classifier
clf1 = RandomForestClassifier()

# Fitting the model using training data