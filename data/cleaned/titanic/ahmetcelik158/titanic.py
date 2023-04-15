#!/usr/bin/env python
# coding: utf-8

# # Titanic - Machine Learning from Disaster
# Hello everyone,
# 
# I am new to meachine learning and would like to try Titanic problem. So I'm welcome to any comment and feedback.
# 
# My plan is to first understand the data with some visualization, then process the data for modeling and finally creating a ML model for prediction.
# 
# **Best Score: 0.79186 - Top %7**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# In[ ]:


df_train = pd.read_csv("data/input/titanic/train.csv")
df_test = pd.read_csv("data/input/titanic/test.csv")
df_train.head()


# # 0. Reference Score

# [Alexi Cook's Titanic Tutorial notebook](https://www.kaggle.com/alexisbcook/titanic-tutorial) is a great tutorial for how to use Kaggle, approach Titanic problem and create a basic ML model and make a prediction. Thanks for the tutorial!
# 
# First, I would like to use the same code from tutorial and make a prediction. So that, i can see how Random Forest model performs and use that score as a benchmark.

# In[ ]:


train_data = pd.read_csv("data/input/titanic/train.csv")
test_data = pd.read_csv("data/input/titanic/test.csv")

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)