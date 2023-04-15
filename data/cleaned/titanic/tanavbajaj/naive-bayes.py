#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes
# 
# It is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
# 
# Let's dive a little into the maths behind Naive Bayes. 
# 
# Starting with the **Bayes theorem**
# 
# ![image.png](attachment:40e3e141-8822-472e-9f0b-5d9ad03085e7.png)
# 
# Here X represents the independent variables while y represents the output or dependent variable. The assumption works that all variables are completely independent of each other hence X translates to x1, x2 x3 and so on
# 
# ![image.png](attachment:45649cac-c917-4319-9a68-9e4daea99c94.png)
# 
# So, the proportionality becomes 
# 
# ![image.png](attachment:dad63e9b-6580-4722-93d4-593f4bad056d.png)
# 
# Combining this for all values of x
# 
# ![image.png](attachment:6c457a6d-0027-448b-8961-3038fc6b6ede.png)
# 
# Now the target for the Naive Bayes algorithm is to find the class which has maximum probability for the target. Which refers to finding the maximum value of y. For this argmax operation is used. 
# 
# ![image.png](attachment:dde1e786-3cac-40ef-9d2e-bef1e1db746a.png)
# 
# 

# # Importing the libraries 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB


# # Reading the dataset

# In[ ]:


dataframe = pd.read_csv("data/input/titanic/train.csv")
test_dataframe = pd.read_csv("data/input/titanic/test.csv")
passangerId = test_dataframe["PassengerId"]


# # Taking only the independent and useful data into the final data frame

# ## Name, Ticket , Passanger ID have almost no correlation to the outcome

# In[ ]:


final_dataframe= dataframe[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']]
final_dataframe = final_dataframe.dropna()
final_dataframe.head()


# Labeling the values in the “Sex” column of the dataset to numbers

# In[ ]:


final_dataframe["Sex"] = final_dataframe["Sex"].replace(to_replace=final_dataframe["Sex"].unique(), value = [1 , 0])


# # One hot encoding
# This is an encoding algorithm in the sklearn library to get categorical data into various columns and make encode it in a way that the dataset can be sent to the machine learning model 

# In[ ]:


final_dataframe = pd.get_dummies(final_dataframe, drop_first=True) 


# # Creating the training and testing datasets

# In[ ]:


train_y = final_dataframe["Survived"]
train_x = final_dataframe[['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked_Q','Embarked_S']]


# In[ ]:


from sklearn.model_selection import train_test_split

train_data, val_data, train_target, val_target = train_test_split(train_x,train_y, train_size=0.8)


# In[ ]:


model = GaussianNB()