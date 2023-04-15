#!/usr/bin/env python
# coding: utf-8

# 
# ![titanic.jpg](attachment:5b38db78-b195-4d7f-987c-b59282135e28.jpg)
# 
# # Titanic - Machine Learning from Disaster
# ## Overview
# 
# This Notebook will be completed in two main ways.<br/>
# First, find and visualize useful data or meaningful relationships within the data.<br/>
# Second, select a model based on the visualization of the previous process. Transform or refine the data into the appropriate form for the model to be used.<br/><br/>
# 
# 
# This competition predicts survival through Survival, Pclass, Sex, Age, ...etc.<br/>
# It is the most representative competition of kaggle, and I will complete my notebooks based on my experience of participating in other competitions.
# 
# ##### "What we need to be careful about here is that we don't have to use all the data to make predictions."<br/>
# 
# #### My opinion :
# * 1) We think it is important to understand the data well during the competition and to select the necessary data well.
# * 2) In addition, the process of preprocessing the data so that the model can learn well is also important.

# ***
# 
# ## My workflow
# #### 1. Import & Install libray
# * Import basic libray
# * Import enginnering libray
# 
# #### 2. Check out my data
# * Check Shape / Info / Describe
# 
# #### 3. Exploratory Data Analysis(EDA) with Visualization [Before Preprocessing]
# * Plot the null values
# * Plot the Distribution of Survived by Sex
# * Plot Crosstab DataFrame ( Survived by Sex | Survived by Pclass )
# * Plot relationship Age and Survived with Violin plot 
# * Plot the Survived and Pclass per Age with Sex [3d interactive Plot]
# * Titanic data Heatmap Plot
# 
# #### 4. Prepocessing Data
# * Null value preprocessing
# * Normalize "Fare" | "Cabin" data
# * Drop unuseful columns
# * Skeweness Value(Outlier) Preprocessing
# 
# #### 5. Feature Enginnering 
# * OneHot Encoding
# * Split Train data / Test data
# 
# #### 6. Modeling
# * LogisticRegression Modeling
# * DecisionTreeClassifier Modeling (with GridSearchCV / CrossValScore)
# * RandomForestClassifier Modeling (with GridSearchCV)
# * LGBMClassifier Modeling (with GridSearchCV)
# * Select Model
# 
# #### 7. Submission
# * Test data Preprocessing
# * Submit the predictions

# # 1. Import & Install libray
# * Import basic libray
# * Import enginnering libray

# In[ ]:


import os
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn')



# In[ ]:


import sklearn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


# In[ ]:


train_df = pd.read_csv("data/input/titanic/train.csv")
test_df = pd.read_csv('data/input/titanic/train.csv')

train_df.head(5)


# # 2. Check out my data
# * Check Shape / Info / Describe

# In[ ]:


print("Titanic train dateset Shape : ", train_df.shape)
print("Titanic test dateset Shape : ", test_df.shape)


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe()


# In[ ]:


Purples_palette = sns.color_palette("Purples", 10)
BuPu_palette = sns.color_palette("BuPu", 10)
sns.palplot(Purples_palette)
sns.palplot(BuPu_palette)


# #### ✔️ This notebook will use this palettes.

# # 3. Exploratory Data Analysis(EDA) with Visualization [Before Preprocessing]
# * Plot the null values
# * Plot the Distribution of Survived by Sex
# * Plot Crosstab DataFrame ( Survived by Sex | Survived by Pclass )
# * Plot relationship Age and Survived with Violin plot 
# * Plot the Survived and Pclass per Age with Sex [3d interactive Plot]
# * Titanic data Heatmap Plot

# ### 3-1) Plot the null values

# In[ ]:


train_df_null_count = pd.DataFrame(train_df.isnull().sum(), columns=["Train Null count"])
test_df_null_count = pd.DataFrame(test_df.isnull().sum(), columns=["Test Null count"])

null_df = pd.concat([train_df_null_count,test_df_null_count],axis=1)
null_df.head(100).style.background_gradient(cmap='Purples')


# In[ ]:


msno.matrix(df=train_df.iloc[:,:],figsize=(5,5),color=BuPu_palette[4])



# ### 3-2) Plot the Distribution of Survived by Sex

# In[ ]:


Purples_palette_two = [Purples_palette[3], Purples_palette[6]]


# In[ ]:


fig = plt.figure(figsize=(8, 8))

gs = fig.add_gridspec(3, 2)


ax_sex_survived = fig.add_subplot(gs[:2,:2])
sns.countplot(x='Sex',hue='Survived', data=train_df, ax=ax_sex_survived, 
              palette=Purples_palette_two)

# ax_survived_sex.set_yticks([])

ax_pie_male = fig.add_subplot(gs[2, 0])
ax_pie_female = fig.add_subplot(gs[2, 1])
# Sex
male = train_df[train_df['Sex']=='male']['Survived'].value_counts().sort_index()
ax_pie_male.pie(male, labels=male.index, autopct='%1.1f%%',explode = (0, 0.1), startangle=90,
               colors=Purples_palette_two
               )

female = train_df[train_df['Sex']=='female']['Survived'].value_counts().sort_index()
ax_pie_female.pie(female, labels=female.index, autopct='%1.1f%%',explode = (0, 0.1), startangle=90,
                colors=Purples_palette_two
                 )

fig.text(0.25,0.92,"Distribution of Survived by Sex", fontweight="bold", fontfamily='serif', fontsize=17)

ax_sex_survived.patch.set_alpha(0)




# ### 3-3) Plot Crosstab DataFrame ( Survived by Sex | Survived by Pclass )

# In[ ]:


pd.crosstab(train_df['Sex'],train_df['Survived'],margins=True).style.background_gradient(cmap='Purples')


# In[ ]:


pd.crosstab(train_df['Pclass'],train_df['Survived'],margins=True).style.background_gradient(cmap='BuPu')


# ### 3-4) Plot relationship Age and Survived with Violin plot 

# In[ ]:


BuPu_palette


# In[ ]:


Purples_palette_two_1 = [Purples_palette[4], Purples_palette[8]]
BuPu_palette_two = [BuPu_palette[2],BuPu_palette[4]]


# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(16,8))


sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train_df, palette=BuPu_palette_two, ax=ax[0])
ax[0].patch.set_alpha(0)
ax[0].text(-0.5,100,"Plot showing the relationship \nbetween Pclass and Age and Survived", fontweight="bold", fontfamily='serif', fontsize=13)


sns.violinplot(x="Sex", y="Age", hue="Survived", data=train_df, palette=Purples_palette_two_1, ax=ax[1])
ax[1].set_yticks([])
ax[1].set_ylabel('')
ax[1].patch.set_alpha(0)
ax[1].text(-0.5,100,"Plot showing the relationship \nbetween Sex and Age and Survived", fontweight="bold", fontfamily='serif', fontsize=13)

fig.text(0.1,1,"Violin plot showing the relationship Age and Survived", fontweight="bold", fontfamily='serif', fontsize=20)



# In[ ]:


train_df


# ### 3-5) Plot the Survived and Pclass per Age with Sex [3d interactive Plot]

# In[ ]:


fig = px.scatter_3d(train_df[:1000], x='Age', y='Survived', z='Pclass',color='Age')
fig.show()


# ### 3-6) Titanic data Heatmap Plot

# In[ ]:


corr = train_df.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, cmap='BuPu')
plt.title("Titanic train data Heatmap", fontweight="bold", fontsize=17)



# # 4. Prepocessing Data
# * Null value preprocessing
# * Normalize "Fare" | "Cabin" data
# * Drop unuseful columns
# * Skeweness Value(Outlier) Preprocessing

# ### 4-1) Null value preprocessing

# In[ ]:


train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Cabin'] = train_df['Cabin'].fillna("N")
train_df['Embarked'] = train_df['Embarked'].fillna("N")


# In[ ]:


train_df.head()


# ### 4-2) Normalize "Fare" | "Cabin" data

# In[ ]:


train_df.loc[train_df["Fare"].isnull(),"Fare"] = train_df["Fare"].mean()
train_df["Fare"] = train_df["Fare"].map(lambda i : np.log(i) if i > 0 else 0)


# ##### => Takes log transformation for data normalization.

# In[ ]:


f, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.distplot(train_df['Fare'], label='Skewness : {:.2f}'.format(train_df['Fare'].skew()), ax=ax, color=BuPu_palette[-1])
plt.legend(loc='best')
plt.title("Check train data Skewness", fontweight="bold", fontsize=18)
ax.patch.set_alpha(0)



# In[ ]:


train_df['Cabin'] = train_df['Cabin'].str[:1]
train_df.head()


# ### 4-3) Drop unuseful columns

# In[ ]:


train_df.drop(['Name','PassengerId','Ticket'],axis=1,inplace=True)
train_df.head()


# In[ ]:


train_df['Cabin'].value_counts()


# In[ ]:


Cabin_T_index = train_df[train_df['Cabin']=='T'].index
train_df.drop(Cabin_T_index,inplace=True)


# In[ ]:


train_df['Embarked'].value_counts()


# In[ ]:


Embarked_N_index = train_df[train_df['Embarked']=='N'].index
train_df.drop(Embarked_N_index,inplace=True)


# In[ ]:


train_df.head()


# # 5. Feature Enginnering 
# * OneHot Encoding
# * Split Train data / Test data

# ### 5-1) OneHot Encoding

# In[ ]:


train_df = pd.get_dummies(train_df)
train_df.head()


# In[ ]:


train_df.shape


# ### 5-2) Split Train data / Test data

# In[ ]:


x = train_df.drop('Survived',axis=1)
y = train_df['Survived']


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[ ]:


print("X train data size : {}".format(x_train.shape))
print("Y train data size : {}".format(y_train.shape))
print(" ")
print("X test data size : {}".format(x_test.shape))
print("Y test data size : {}".format(y_test.shape))


# # 6. Modeling
# * LogisticRegression Modeling
# * DecisionTreeClassifier Modeling (with GridSearchCV / CrossValScore)
# * RandomForestClassifier Modeling (with GridSearchCV)
# * LGBMClassifier Modeling (with GridSearchCV)
# * Select Model

# ### 6-1) LogisticRegression Modeling

# In[ ]:


log_reg = LogisticRegression()