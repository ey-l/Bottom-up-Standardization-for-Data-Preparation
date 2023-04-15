#!/usr/bin/env python
# coding: utf-8

# <div style="color:black;
#            display:fill;
#            border-radius:1px;
#            background-color:#87ceeb;
#            align:center;">
# 
# <p style="padding: 1px;
#               color:white;">
# <center>
#     
# # Understanding Pipeline in sklearn

# ## Table of Contents
# 
# * [1. Introduction to Pipeline](#1)
# * [2. Examples](#2)
#      - [2.1 : Example1: Linear Regression on Sinusoids](#3)
#      - [2.2 : Example2: Cancer dataset](#4)  
#      - [2.3 : Example3: Titanic dataset](#5)  
#        - [2.3.1 : Library and data import](#6)
#        - [2.3.2 : Define pipelines](#7)
#        - [2.3.3 : Define final pipeline and predict](#8)
# * [3. References](#9)

# <div style="color:black;
#            display:fill;
#            border-radius:1px;
#            background-color:#e4f2f8;
#            align:center;">
# 
# <p style="padding: 1px;
#               color:white;">
# <center>
#     
# <a id="1"></a>
# ## 1. Introduction to Pipeline

# Machine Learning problem commonly involves two steps. First, we sequentially transform the data comprising of several steps such as feature transformation,dimensionality reduction, standardization etc. Secondly, we learn from the data by using an estimator or regressor to gain insights.
# 
# Pipeline simplifies the use of Machine learning model by combining various data transformation part with the data estimation part into single unit.In this notebook, We will illustrate the use of pipeline in Sci-Kit Learn library through examples.
# 
# 
# ![image.png](attachment:60655edc-141c-4c7e-8211-37ce4af86a24.png)

# <div style="color:black;
#            display:fill;
#            border-radius:1px;
#            background-color:#e4f2f8;
#            align:center;">
# 
# <p style="padding: 1px;
#               color:white;">
# <center>
#     
# <a id="2"></a>
# ## 2. Examples

# <a id="3"></a>
# ### 2.1: Example1: Linear Regression on Sinusoids 

# ##### Generate Training Data
# 
# - For the training data, we will use single feature as input and the response is a      
#   sinusoidal function of input feature (Y = X + sin(X))
# - We will add some noise to the response to make it realistic for later predictions

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
X_train = np.linspace(0, 10 * np.pi, num=1000); # input feature
noise = np.random.normal(scale=3, size=X_train.size); # additive noise
y_train = X_train + 10 * np.sin(X_train)+ noise; # response variable
plt.scatter(X_train,y_train,color='g'); # visualize the dataset
plt.xlabel('training feature');
plt.ylabel('training response');


# ##### Generate Test Data

# In[ ]:


X_test = np.linspace(10, 15 * np.pi, num=100); # test data is again a linear array
noise = np.random.normal(scale=3, size=X_test.size); # we add noise sameway as we did for train
y_true = X_test + 10 * np.sin(X_test)+ noise; # true response desired from test data
plt.scatter(X_test,y_true,color='g'); # visualize test feature and test response
plt.xlabel('test feature');
plt.ylabel('test response');


# ##### Linear Regression (without using feature engineering)
# 
# - Now let us blindly use linear regression on the example1 data to
#   see how it fits the data

# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression();