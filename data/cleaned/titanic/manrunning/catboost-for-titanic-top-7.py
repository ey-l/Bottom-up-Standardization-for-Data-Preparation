#!/usr/bin/env python
# coding: utf-8

# Hi, every one. I am almost new to the kaggle, and I do the ML and DL jobs during my work. I find the kaggle as best for me to make it done. With no more words , I used the catboost and Tensorflow to make the Titanic prediction, and I got the top 7% for it, and I want to share with others what I have done and if is there any better suggestion for it , I want to discuss with you! Here we go!

# In[1]:


#import 
import numpy as np
import pandas as pd
import hyperopt
from catboost import Pool, CatBoostClassifier, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


#get the train and test data
train_df = pd.read_csv('data/input/titanic/train.csv')
test_df = pd.read_csv('data/input/titanic/test.csv')


# In[3]:


#show the train data
train_df.info()


# In[4]:


#show how many the null value for each column
train_df.isnull().sum()


# In[5]:


#for the train data ,the age ,fare and embarked has null value,so just make it -999 for it
#and the catboost will distinguish it
train_df.fillna(-999,inplace=True)
test_df.fillna(-999,inplace=True)


# In[6]:


#now we will get the train data and label
x = train_df.drop('Survived',axis=1)
y = train_df.Survived


# In[7]:


#show what the dtype of x, note that the catboost will just make the string object to categorical 
#object inside
x.dtypes


# In[8]:


#choose the features we want to train, just forget the float data
cate_features_index = np.where(x.dtypes != float)[0]


# In[9]:


#make the x for train and test (also called validation data) 
xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=.85,random_state=1234)


# In[10]:


#let us make the catboost model, use_best_model params will make the model prevent overfitting
model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)


# In[11]:


#now just to make the model to fit the data