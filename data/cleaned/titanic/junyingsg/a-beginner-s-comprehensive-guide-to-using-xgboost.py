#!/usr/bin/env python
# coding: utf-8

# # The Challenge
# 
# The sinking of the Titanic is one of the most infamous shipwrecks in history.
# 
# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# 
# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# 
# In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
# 
# 

# ![](http://cdn.britannica.com/72/153172-050-EB2F2D95/Titanic.jpg)

# # Importing libraries and data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df_train = pd.read_csv("data/input/titanic/train.csv")
df_train.head()


# In[ ]:


df_train.drop(["PassengerId", "Ticket"], axis=1, inplace=True)
df_train.describe()


# From this, we can draw several quick observations:
# 
# 1. Roughly 38% of passengers survived
# 
# 
# 2. The average passenger is around 30 years old, but there is high variance as standard deviation is >14.5. There are babies and elderly onboard as the minimum age is <1 and the oldest person onboard is 80 years old. However, most passengers (up to the 75th percentile) are <39 years old.
# 
# 
# 3. Most people only brought 1 spouse/sibling onboard (up to 75th percentile). However, there are a few who brought a large family (up to 8 spouse + siblings). 
# 
# 
# 4. Most people did not bring their parents or children onboard (up to 75th percentile). But again, there are a few who brought a large family (up to 6 parents + children). 
# 
# 
# 5. Most people paid a relatively low fare (lower than the mean) for their tickets (75% paid 31 dollars or less while the mean is 32.2 dollars). However, there are a few passengers who paid an exorbitant price for their tickets (up to 512 dollars), possibly indicating the presence of VIPs.

# In[ ]:


df_train.isnull().sum()


# # Dealing with missing values

# First, we are going to deal with the missing "Age" data. There are a few ways to impute the data, most commonly using the mean or the median. However, the mean is easily affected by extreme values and while the median is generally a good representation of our age distribution, we want a value that can represent the different demographics of our passengers. Hence, we will use salutations (Mr, Ms, Mdm etc) as a separator for the different demographics.
# 
# Let's first extract the salutations from our dataset.

# In[ ]:


import re
df_train["Salutations"] = df_train["Name"].str.extract(r'([A-Z]{1}[a-z]+\.)')


# In[ ]:


df_train["Salutations"].unique()


# From salutations alone, we managed to glean even more interesting data. It seems that there are nobility (Don, Countess and Jonkheer) and military personnel (Major, Col, Capt) onboard. There are also French equivalent of English salutations (Mme = Mrs, Mlle = Ms) and clergymen present(Rev, possibly Don). Let's further explore the salutations that are age significant (Master, Miss, Mister, Mrs).

# In[ ]:


df_train[df_train["Salutations"] == "Master."]["Age"].median()


# In[ ]:


df_train[(df_train["Salutations"] == "Miss.") | (df_train["Salutations"] == "Ms.") | (df_train["Salutations"] == "Mlle.")]["Age"].median()


# In[ ]:


df_train[df_train["Salutations"] == "Mr."]["Age"].median()


# In[ ]:


df_train[(df_train["Salutations"] == "Mrs.") | (df_train["Salutations"] == "Mme.")]["Age"].median()


# As we can see from their median values, each salutation represents a different age group. It is also reasonable to assume that high-ranking military personnel and nobility are usually older, so we shall group them along with other uncommon titles such as Rev and Dr. Let's now create a new feature column with these categories.

# In[ ]:


master = (df_train["Salutations"] == "Master.")
miss = (df_train["Salutations"] == "Miss.") | (df_train["Salutations"] == "Ms.") | (df_train["Salutations"] == "Mlle.")
mister = (df_train["Salutations"] == "Mr.")
missus = (df_train["Salutations"] == "Mrs.") | (df_train["Salutations"] == "Mme.")


# In[ ]:


df_train["Title"] = "Others"
df_train["Title"][master] = "Master"
df_train["Title"][miss] = "Miss"
df_train["Title"][mister] = "Mister"
df_train["Title"][missus] = "Missus"


# We will now fill in the missing values for "Age" according to their titles. We will use the median for each demographic as a replacement.

# In[ ]:


df_train["Age"] = df_train.groupby("Title")["Age"].apply(lambda x: x.fillna(x.median()))


# Now we have to deal with the missing data for "Cabin". A large proportion of it is missing, but the little amount of data that we have is important as the first letter of each cabin represents the deck. 'A' is at the top, 'B' is below 'A' and so on. We will replace all missing values with 'Z' and come back to analyze this later on.

# In[ ]:


df_train["Cabin"] = df_train["Cabin"].str.extract(r'([A-Z]{1})').fillna('Z')


# Finally, we have to deal with 2 missing values in "Embarked". Let's just replace them with the most common value.

# In[ ]:


df_train["Embarked"].mode()


# In[ ]:


df_train["Embarked"] = df_train["Embarked"].fillna('S')


# Now, let's check our dataset before we start analysing the data.

# In[ ]:


df_train.info()


# Great. We managed to replace all missing values. Let's drop the irrelevant features in our dataset and take a final look at our clean data.

# In[ ]:


df_train.drop(["Name", "Salutations"], axis=1, inplace=True)
df_train.head()


# # Data Visualisation & Feature Engineering

# We will take a look at each feature and see how they relate to survival.

# In[ ]:


plt.figure(figsize=(8,8))
sns.countplot("Pclass", data=df_train, hue="Survived")


# Next, let's take a look at sex and see how it affects survival.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 6))
ax=sns.countplot("Sex", data=df_train, hue="Survived", ax = axes[0])
ax1=df_train["Sex"].value_counts().plot.pie(autopct='%1.1f%%', ax = axes[1])


# It is evident that females have a higher rate of survival as compared to men (65% of passengers were male and most of them did not survive). Let's convert this categorical data to numerical. 1 for female, and 0 for men.

# In[ ]:


sex = {"male": 0, "female": 1}
df_train["Sex"] = df_train["Sex"].map(sex)


# Next, we want to see how age affects chances of survival. This is especially important as we previously identified that there is high variance in age, that 75% of passengers are <39 years old, and there are babies and elderly onboard. We want to see how those factors are related to survival. Let's first plot the distribution of age against survival.

# In[ ]:


plt.figure(figsize=(8,8))
survived = df_train[df_train["Survived"] == 1]
not_survived =  df_train[df_train["Survived"] == 0]

sns.distplot(survived["Age"], kde=False, label='Survived')
sns.distplot(not_survived["Age"], kde=False, label='Did not survive')
plt.legend()


# It is difficult to relate survival with age alone as they seem to follow the same distribution. Let's try including sex as a factor as we saw that majority of survivors were female.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12, 8))
men = df_train[df_train["Sex"] == 0]
women = df_train[df_train["Sex"] == 1]

ax = sns.distplot(women[women['Survived']==1]["Age"], label="Survived", ax=axes[0], kde=False, bins=20)
ax1 = sns.distplot(women[women['Survived']==0]["Age"], label="Did not survive", ax=axes[0], kde=False, bins=20)
ax.legend()
ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1]["Age"], label="Survived", ax=axes[1], kde=False, bins=20)
ax1 = sns.distplot(men[men['Survived']==0]["Age"], label="Did not survive", ax=axes[1], kde=False, bins=20)
ax.legend()
ax.set_title('Male')


# Awesome. Now we can see that majority of both genders age <8 has a high survival rate, women have a high survival rate between age 14 - 40, and men aged slightly <30 have the highest risk of dying. Let's translate age into a equal-sized feature column with the help of qcut function from pandas.

# In[ ]:


pd.qcut(df_train[df_train["Age"]>8]["Age"], 5)


# In[ ]:


age = (df_train["Age"] < 8)
age1 = (df_train["Age"] >= 9) & (df_train["Age"] < 21)
age2 = (df_train["Age"] >= 21) & (df_train["Age"] < 28)
age3 = (df_train["Age"] >= 28) & (df_train["Age"] < 30)
age4 = (df_train["Age"] >= 28) & (df_train["Age"] < 39)
age5 = (df_train["Age"] >= 39)


# In[ ]:


df_train["Age"][age] = 0
df_train["Age"][age1] = 1
df_train["Age"][age2] = 2
df_train["Age"][age3] = 3
df_train["Age"][age4] = 4
df_train["Age"][age5] = 5


# Next, we will combine the number of sibilings/spouses and parents/children to get the total number of family members onboard. Then we will take a look at how that relates to survival.

# In[ ]:


plt.figure(figsize=(10, 10))
df_train["Family"] = df_train["SibSp"] + df_train["Parch"]
survived = df_train[df_train["Survived"] == 1]
not_survived =  df_train[df_train["Survived"] == 0]

sns.distplot(survived["Family"], kde=False, label="Survived", bins=50)
sns.distplot(not_survived["Family"], kde=False, label="Did not survive", bins=50)
plt.legend()


# From here, we can see that your chances of survival is lowest when you have no family and have 4 or more family members. Let's reflect that as a feature column.

# In[ ]:


none = (df_train["Family"] == 0)
four = (df_train["Family"] >= 4)

df_train["Fam_Cat"] = 1
df_train["Fam_Cat"][none] = 0
df_train["Fam_Cat"][four] = 2


# Now we will look at how Fare paid affects your chances of survival.

# In[ ]:


plt.figure(figsize=(10, 10))
survived = df_train[df_train["Survived"] == 1]
not_survived =  df_train[df_train["Survived"] == 0]

sns.distplot(survived["Fare"], kde=False, label='Survived', bins=100, color='green')
sns.distplot(not_survived["Fare"], kde=False, label='Did not survive', bins=100, color='red')
plt.legend()


# From here, we can see that if you paid less, your relative chances of survival decreased. This is especially so when you paid <10, and between 10 - 20. Your chances of survival increased when you paid more than $50. Let's classify this into equal categories (like age).

# In[ ]:


pd.qcut(df_train["Fare"], 6)


# In[ ]:


fare = (df_train["Fare"] < 7.775)
fare1 = (df_train["Fare"] >= 7.775) & (df_train["Fare"] < 8.662)
fare2 = (df_train["Fare"] >= 8.662) & (df_train["Fare"] < 14.454)
fare3 = (df_train["Fare"] >= 14.454) & (df_train["Fare"] < 26)
fare4 = (df_train["Fare"] >= 26) & (df_train["Fare"] < 52.369)
fare5 = (df_train["Fare"] >= 52.369)

df_train["Fare"][fare] = 0
df_train["Fare"][fare1] = 1
df_train["Fare"][fare2] = 2
df_train["Fare"][fare3] = 3
df_train["Fare"][fare4] = 4
df_train["Fare"][fare5] = 5


# Now, let's analyze the port of embarkation.

# In[ ]:


sns.catplot(x='Sex', y='Survived', kind='bar', data=df_train, hue='Embarked', palette='rocket', aspect=1.3)


# Survival rate seems to be highest if embarked from C, and uncertain if embarked from S and Q. Since there are only 3 ports, let's convert this to numeric values.

# In[ ]:


port = {'S': 1, 'C': 2, 'Q': 3}
df_train["Embarked"] = df_train["Embarked"].map(port)


# Finally, let's convert Cabin categories to numbers. According to the diagram below, A is nearest to the top, hence we can assume that passengers in Cabin A have the highest chance of survival, followed by B and then C and so on. We won't analyse this as there is too much missing data to draw conclusions about the overall population of passengers.

# ![](https://upload.wikimedia.org/wikipedia/commons/8/84/Titanic_cutaway_diagram.png)

# In[ ]:


df_train["Cabin"].unique()


# In[ ]:


cabin = {"Z": 8, "T": 7, "G": 6, "F": 5, "E": 4, "D": 3, "C": 2, "B": 1, "A": 0}
df_train["Cabin"] = df_train["Cabin"].map(cabin)


# Let's drop all irrelevant columns and prep our data for model building.

# In[ ]:


df_train.head()


# In[ ]:


df = df_train.drop(["SibSp", "Parch", "Family"], axis=1)
df.head()


# Let's map our "Title" category and create a few final features for Pclass, Sex, Age and Fare as these seem to be the most important features.

# In[ ]:


title = {"Master": 0, "Miss": 1, "Mister": 3, "Missus": 4, "Others": 5}
df["Title"] = df["Title"].map(title)

df["Age"] = df["Age"].astype(int)
df["Fare"] = df["Fare"].astype(int)

df["Pclass_Sex"] = df["Pclass"]*df["Sex"]
df["Pclass_Age"] = df["Pclass"]*df["Age"]
df["Pclass_Fare"] = df["Pclass"]*df["Fare"]
df["Sex_Age"] = df["Sex"]*df["Age"]
df["Sex_Fare"] = df["Sex"]*df["Fare"]
df["Age_Fare"] = df["Age"]*df["Fare"]


# In[ ]:


df.head()


# # Model Building and Training

# In this section, we will be using an algorithm known as eXtreme Gradient Boosting (XGBoost). It is somewhat similar to a Random Forest Classifier, but trains decision trees sequentially (one at a time) and each tree is design to rectify errors made by the previous tree through gradient descent. It is one of the most well known classification algorithms for its performance and speed.

# We will split our train data into two different sets of data. 80% of it is for training our model, 20% of it is set aside for validation (to see how our model generalizes to new data).

# In[ ]:


X = df.iloc[:, 1:]
y = df.iloc[:, 0]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)


# Now, we will manually select parameters based on commonly used values.

# In[ ]:


from xgboost import XGBClassifier
classifier = XGBClassifier(nthread=1, colsample_bytree=0.8, learning_rate=0.03, max_depth=4, min_child_weight=2, n_estimators=1000, subsample=0.8)