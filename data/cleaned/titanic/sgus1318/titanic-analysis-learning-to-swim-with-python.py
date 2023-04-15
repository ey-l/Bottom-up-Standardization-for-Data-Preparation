#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# 
# This is my first Kaggle, and my first foray into data analysis using python.  The following kernel contains the steps enumerated below for assessing the Titanic survival dataset:
# 
# 1. Import data and python packages
# 2. Assess Data Quality & Missing Values
#     * 2.1 Age - Missing Values
#     * 2.2 Cabin - Missing Values
#     * 2.3 Embarked - Missing Values
#     * 2.4 Final Adjustments to Data
#     * 2.4.1 Additional Variables
# 3. Exploratory Data Analysis
# 4. Logistic Regression 
# 5. Hold-Out Testing & Model Assessment
#     * 5.1 Kaggle "Test" Dataset
#     * 5.2 Re-run Logistic Regression w/ 80-20 Split
#     * 5.3 Out-of-sample test results
# 6. Logistic Regression Conclusions<br>
# 7. Alternate Approach: Random Forest Estimation
# <br>
# *References are provided at the bottom of each section.*

# ## 1. Import Data & Python Packages

# In[ ]:


import numpy as np 
import pandas as pd 

from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)

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


# <font color=red>  Note: There is no target variable for the hold out data (i.e. "Survival" column is missing), so there's no way to use this as our cross validation sample.  Refer to Section 5.</font>

# ## 2. Data Quality & Missing Value Assessment

# In[ ]:


# check missing values in train dataset
titanic_df.isnull().sum()


# ### 2.1    Age - Missing Values

# In[ ]:


sum(pd.isnull(titanic_df['Age']))


# In[ ]:


# proportion of "Age" missing
round(177/(len(titanic_df["PassengerId"])),4)


# ~20% of entries for passenger age are missing. Let's see what the 'Age' variable looks like in general.

# In[ ]:


titanic_df["Age"].hist(bins=15, color='teal', alpha=0.8)


# Since "Age" is (right) skewed, using the mean might give us biased results by filling in ages that are older than desired.  To deal with this, we'll use the median to impute the missing values. 

# In[ ]:


# median age is 28 (as compared to mean which is ~30)
titanic_df["Age"].median(skipna=True)


# ### 2.2 Cabin - Missing Values

# In[ ]:


# proportion of "cabin" missing
round(687/len(titanic_df["PassengerId"]),4)


# 77% of records are missing, which means that imputing information and using this variable for prediction is probably not wise.  We'll ignore this variable in our model.

# ### 2.3 Embarked - Missing Values

# In[ ]:


# proportion of "Embarked" missing
round(2/len(titanic_df["PassengerId"]),4)


# There are only 2 missing values for "Embarked", so we can just impute with the port where most people boarded.

# In[ ]:


sns.countplot(x='Embarked',data=titanic_df,palette='Set2')



# By far the most passengers boarded in Southhampton, so we'll impute those 2 NaN's w/ "S".

# *References for graph creation:*<br>
# https://matplotlib.org/1.2.1/examples/pylab_examples/histogram_demo.html <br>
# https://seaborn.pydata.org/generated/seaborn.countplot.html

# ### 2.4 Final Adjustments to Data (Train & Test)

# Based on my assessment of the missing values in the dataset, I'll make the following changes to the data:
# * If "Age" is missing for a given row, I'll impute with 28 (median age).
# * If "Embark" is missing for a riven row, I'll impute with "S" (the most common boarding port).
# * I'll ignore "Cabin" as a variable.  There are too many missing values for imputation.  Based on the information available, it appears that this value is associated with the passenger's class and fare paid.

# In[ ]:


train_data = titanic_df
train_data["Age"].fillna(28, inplace=True)
train_data["Embarked"].fillna("S", inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)


# ### 2.4.1 Additional Variables

# According to the Kaggle data dictionary, both SibSp and Parch relate to traveling with family.  For simplicity's sake (and to account for possible multicollinearity), I'll combine the effect of these variables into one categorical predictor: whether or not that individual was traveling alone.

# In[ ]:


## Create categorical variable for traveling alone

train_data['TravelBuds']=train_data["SibSp"]+train_data["Parch"]
train_data['TravelAlone']=np.where(train_data['TravelBuds']>0, 0, 1)


# In[ ]:


train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)
train_data.drop('TravelBuds', axis=1, inplace=True)


# I'll also create categorical variables for Passenger Class ("Pclass"), Gender ("Sex"), and Port Embarked ("Embarked"). 

# In[ ]:


#create categorical variable for Pclass

train2 = pd.get_dummies(train_data, columns=["Pclass"])


# In[ ]:


train3 = pd.get_dummies(train2, columns=["Embarked"])


# In[ ]:


train4=pd.get_dummies(train3, columns=["Sex"])
train4.drop('Sex_female', axis=1, inplace=True)


# In[ ]:


train4.drop('PassengerId', axis=1, inplace=True)
train4.drop('Name', axis=1, inplace=True)
train4.drop('Ticket', axis=1, inplace=True)
train4.head(5)


# In[ ]:


df_final = train4


# ### Now, apply the same changes to the test data. <br>
# I will apply to same imputation for "Age" in the Test data as I did for my Training data (if missing, Age = 28).  <br> I'll also remove the "Cabin" variable from the test data, as I've decided not to include it in my analysis. <br> There were no missing values in the "Embarked" port variable. <br> I'll add the dummy variables to finalize the test set.  <br> Finally, I'll impute the 1 missing value for "Fare" with the median, 14.45.

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


# *References for categorical variable creation: <br>
# http://pbpython.com/categorical-encoding.html <br>
# https://chrisalbon.com/python/data_wrangling/pandas_create_column_using_conditional/*

# ## 3. Exploratory Data Analysis

# ## 3.1 Exploration of Age

# In[ ]:


plt.figure(figsize=(15,8))
sns.kdeplot(titanic_df["Age"][df_final.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(titanic_df["Age"][df_final.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')



# The age distribution for survivors and deceased is actually very similar.  One notable difference is that, of the survivors, a larger proportion were children.  The passengers evidently made an attempt to save children by giving them a place on the life rafts. 

# In[ ]:


plt.figure(figsize=(20,4))
avg_survival_byage = round(df_final[["Age", "Survived"]].groupby(['Age'],as_index=False).mean(),1)
g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")


# Considering the survival rate of passengers under 16, I'll also include another categorical variable in my dataset: "Minor"

# In[ ]:


df_final['IsMinor']=np.where(train_data['Age']<=16, 1, 0)


# In[ ]:


final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)


# ## 3.2 Exploration of Fare

# In[ ]:


plt.figure(figsize=(15,8))
sns.kdeplot(df_final["Fare"][titanic_df.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(df_final["Fare"][titanic_df.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
# limit x axis to zoom on most information. there are a few outliers in fare. 
plt.xlim(-20,200)



# As the distributions are clearly different for the fares of survivors vs. deceased, it's likely that this would be a significant predictor in our final model.  Passengers who paid lower fare appear to have been less likely to survive.  This is probably strongly correlated with Passenger Class, which we'll look at next.

# ## 3.3 Exploration of Passenger Class

# In[ ]:


sns.barplot('Pclass', 'Survived', data=titanic_df, color="darkturquoise")



# Unsurprisingly, being a first class passenger was safest.

# ## 3.4 Exploration of Embarked Port

# In[ ]:


sns.barplot('Embarked', 'Survived', data=titanic_df, color="teal")



# Passengers who boarded in Cherbourg, France, appear to have the highest survival rate.  Passengers who boarded in Southhampton were marginally less likely to survive than those who boarded in Queenstown.  This is probably related to passenger class, or maybe even the order of room assignments (e.g. maybe earlier passengers were more likely to have rooms closer to deck). <br> It's also worth noting the size of the whiskers in these plots.  Because the number of passengers who boarded at Southhampton was highest, the confidence around the survival rate is the highest.  The whisker of the Queenstown plot includes the Southhampton average, as well as the lower bound of its whisker.  It's possible that Queenstown passengers were equally, or even more, ill-fated than their Southhampton counterparts.

# ## 3.5 Exploration of Traveling Alone vs. With Family

# In[ ]:


sns.barplot('TravelAlone', 'Survived', data=df_final, color="mediumturquoise")



# Individuals traveling without family were more likely to die in the disaster than those with family aboard.  Given the era, it's likely that individuals traveling alone were likely male.

# ## 3.6 Exploration of Gender Variable

# In[ ]:


sns.barplot('Sex', 'Survived', data=titanic_df, color="aquamarine")



# This is a very obvious difference.  Clearly being female greatly increased your chances of survival.

# References: <br>
# https://seaborn.pydata.org/generated/seaborn.barplot.html <br>
# https://seaborn.pydata.org/generated/seaborn.kdeplot.html

# ## 4. Logistic Regression and Results

# In[ ]:


df_final.head(10)


# In[ ]:


cols=["Age", "Fare", "TravelAlone", "Pclass_1", "Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
X=df_final[cols]
Y=df_final['Survived']


# In[ ]:


import statsmodels.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
logit_model=sm.Logit(Y,X)