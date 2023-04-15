#!/usr/bin/env python
# coding: utf-8

# <center><font size="10">🚢🏊🏻‍♀️Titanic Start Here: A GENTLE Introduction</font></center>
# <br>
# <center><font size="3">Introdution</font></center>
# > In this Kernel we will see 3 approaches to the classification task in detail.
# > 1. [Import Data & Python Packages](#1-bullet) <br>
# > 2. [Missing Value Handling](#2-bullet)<br>
# >     * [2.1 Age - Missing Values](#2.1-bullet) <br>
# >     * [2.2 Embarked - Missing Values](#2.2-bullet) <br>
# >     * [2.3 Final Adjustments to Data](#2.3-bullet) <br>
# >     * [2.4 Additional Variables](#2.4-bullet) <br> 
# > 3. [Exploratory Data Analysis](#3-bullet) <br>
# > 4. [Alternate Approach 1 :Logistic Regression](#4-bullet) <br>
# > 5. [Alternate Approach 2 : Random Forest Estimation](#5-bullet) <br>
# > 6. [Alternate Approach 3: Decision Tree](#6-bullet) <br>
# > 7. [Ensemble](#7-bullet)
# > 8. [TOP 1% Solution GA](#8-bullet)

# ## 1. Import Data & Python Packages <a class="anchor" id="1-bullet"></a>

# In[ ]:


import numpy as np 
import pandas as pd 
import sys

from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="dark") #white background style for seaborn plots
sns.set(style="darkgrid", color_codes=True)
RED   = "\033[1;31m"  
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
#sklearn imports source: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8


# In[ ]:


# get titanic & test csv files as a DataFrame

#developmental data (train)
titanic_df = pd.read_csv("data/input/titanic/train.csv")

#cross validation data (hold-out testing)
test_df    = pd.read_csv("data/input/titanic/test.csv")

# preview developmental data
titanic_df.head(5)


# In[ ]:


test_df.head(5)


# ## 2. Data Quality & Missing Value Assessment <a class="anchor" id="2-bullet"></a>

# ### 2.1    Age - Missing Values <a class="anchor" id="2.1-bullet"></a>

# In[ ]:


a=sum(pd.isnull(titanic_df['Age'])) # COUNT Missing Values in age
b=round(a/(len(titanic_df["PassengerId"])),4) # proportion of "Age" missing in percent
sys.stdout.write(GREEN)
print("Count of missing Values : {} , The Proportion of this values with dataset is {}\n".format(a,b*100))
sys.stdout.write(CYAN)
print("visualization AGE")
ax = titanic_df["Age"].hist(bins=15, color='#34495e', alpha=0.9)
ax.set(xlabel='Age', ylabel='Count')



# > Since "Age" is (right) skewed, using the mean might give us biased results by filling in ages that are older than desired.  To deal with this, we'll use the median to impute the missing values. 

# In[ ]:


m1=titanic_df["Age"].median(skipna=True)
m2=titanic_df["Age"].mean(skipna=True)
sys.stdout.write(CYAN)
print("Median: {} and Mean: {} | Median age is 28 as compared to mean which is ~30".format(m1,m2))


# ### 2.2 Embarked - Missing Values <a class="anchor" id="2.2-bullet"></a>

# In[ ]:


# proportion of "Embarked" missing
a=round(2/len(titanic_df["PassengerId"]),4)
sys.stdout.write(CYAN)
print('proportion of "Embarked" missing is {}'.format(a*100))


# In[ ]:


sys.stdout.write(CYAN)
print('visualization Embarked')
sns.countplot(x='Embarked',data=titanic_df,palette='Set1')



# ### 2.3 Final Adjustments to Data (Train & Test) <a class="anchor" id="2.3-bullet"></a>
# 
# > Based on my assessment of the missing values in the dataset, I'll make the following changes to the data:
# > * If "Age" is missing for a given row, I'll impute with 28 (median age).
# > * If "Embark" is missing for a riven row, I'll impute with "S" (the most common boarding port).
# > * I'll ignore "Cabin" as a variable.  There are too many missing values for imputation.  Based on the information available, it appears that this value is associated with the passenger's class and fare paid.

# In[ ]:


train_data = titanic_df
train_data["Age"].fillna(28, inplace=True)
train_data["Embarked"].fillna("S", inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)


# ### 2.4 Additional Variables <a class="anchor" id="2.4-bullet"></a>
# 
# > According to the Kaggle data dictionary, both SibSp and Parch relate to traveling with family.  For simplicity's sake (and to account for possible multicollinearity), we will combine the effect of these variables into one categorical predictor: whether or not that individual was traveling alone.

# In[ ]:


## Create categorical variable for traveling alone
train_data['TravelBuds']=train_data["SibSp"]+train_data["Parch"]
train_data['TravelAlone']=np.where(train_data['TravelBuds']>0, 0, 1)

train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)
train_data.drop('TravelBuds', axis=1, inplace=True)

#create categorical variable for Pclass || ONE HOT ENCODING
train2 = pd.get_dummies(train_data, columns=["Pclass"])

train3 = pd.get_dummies(train2, columns=["Embarked"])

train4=pd.get_dummies(train3, columns=["Sex"])
train4.drop('Sex_female', axis=1, inplace=True)

#Drop Unwanted
train4.drop('PassengerId', axis=1, inplace=True)
train4.drop('Name', axis=1, inplace=True)
train4.drop('Ticket', axis=1, inplace=True)
train4.head(5)
df_final = train4


# ### Apply the same changes to the test data. <br>
# 

# In[ ]:


test_df["Age"].fillna(28, inplace=True)
test_df["Fare"].fillna(14.45, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)


# In[ ]:


test_df['TravelBuds']=test_df["SibSp"]+test_df["Parch"]
test_df['TravelAlone']=np.where(test_df['TravelBuds']>0, 0, 1)

test_df.drop('SibSp', axis=1, inplace=True)
test_df.drop('Parch', axis=1, inplace=True)
test_df.drop('TravelBuds', axis=1, inplace=True)

test2 = pd.get_dummies(test_df, columns=["Pclass"])
test3 = pd.get_dummies(test2, columns=["Embarked"])

test4=pd.get_dummies(test3, columns=["Sex"])
test4.drop('Sex_female', axis=1, inplace=True)

test4.drop('PassengerId', axis=1, inplace=True)
test4.drop('Name', axis=1, inplace=True)
test4.drop('Ticket', axis=1, inplace=True)
final_test = test4


# In[ ]:


final_test.head(5)


# ## 3. Exploratory Data Analysis <a class="anchor" id="3-bullet"></a>

# ## 3.1 Exploration of Age <a class="anchor" id="3.1-bullet"></a>

# In[ ]:


sys.stdout.write(GREEN)
print("Density Plot of Age for Surviving Population and Deceased Population")
plt.figure(figsize=(15,8))
sns.kdeplot(titanic_df["Age"][df_final.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(titanic_df["Age"][df_final.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')



# > The age distribution for survivors and deceased is actually very similar.  One notable difference is that, of the survivors, a larger proportion were children.  The passengers evidently made an attempt to save children by giving them a place on the life rafts. 

# In[ ]:


plt.figure(figsize=(25,8))
avg_survival_byage = df_final[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")


# > Considering the survival rate of passengers under 16, I'll also include another categorical variable in my dataset: "Minor"

# In[ ]:


df_final['IsMinor']=np.where(train_data['Age']<=16, 1, 0)


# In[ ]:


final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)


# ## 3.2 Exploration of Fare <a class="anchor" id="3.2-bullet"></a>

# In[ ]:


plt.figure(figsize=(15,8))
sns.kdeplot(df_final["Fare"][titanic_df.Survived == 1], color="#e74c3c", shade=True)
sns.kdeplot(df_final["Fare"][titanic_df.Survived == 0], color="#3498db", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
# limit x axis to zoom on most information. there are a few outliers in fare. 
plt.xlim(-20,200)



# ## 3.3 Exploration of Passenger Class <a class="anchor" id="3.3-bullet"></a>

# In[ ]:


sns.barplot('Pclass', 'Survived', data=titanic_df, color="#2ecc71")



# > Unsurprisingly, being a first class passenger was safest.

# ## 3.4 Exploration of Embarked Port <a class="anchor" id="3.4-bullet"></a>

# In[ ]:


sns.barplot('Embarked', 'Survived', data=titanic_df, color="#2ecc71")



# ## 3.5 Exploration of Traveling Alone vs. With Family <a class="anchor" id="3.5-bullet"></a>

# In[ ]:


sns.barplot('TravelAlone', 'Survived', data=df_final, color="#2ecc71")



# > Individuals traveling without family were more likely to die in the disaster than those with family aboard.  Given the era, it's likely that individuals traveling alone were likely male.

# ## 4. Logistic Regression <a class="anchor" id="4-bullet"></a>

# In[ ]:


cols=["Age", "Fare", "TravelAlone", "Pclass_1", "Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
X=df_final[cols]
Y=df_final['Survived']


# In[ ]:


import statsmodels.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
logit_model=sm.Logit(Y,X)