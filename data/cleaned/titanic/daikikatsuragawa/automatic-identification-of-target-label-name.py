#!/usr/bin/env python
# coding: utf-8

# ## Abstract
# 
# Specific extraction of target label name from the difference between training data and test data columns. I think that it can be used when you want to create a benchmark quickly in a competition with a simple structure such as Titanic.

# In[ ]:


import pandas as pd

train_df = pd.read_csv('data/input/titanic/train.csv')
test_df = pd.read_csv('data/input/titanic/test.csv')
train_columns = train_df.columns.to_list()
test_columns = test_df.columns.to_list()
tmp = list(set(train_columns) - set(test_columns))
label = tmp[0]

print(label)


# ## Example
# 
# 
# ### Example of use
# 
# Here's an example of using AutoGluon for the next notebook.
# 
# https://www.kaggle.com/daikikatsuragawa/a-beginner-s-guide-to-autogluon

# In[ ]:





# In[ ]:


import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

train= TabularDataset('data/input/titanic/train.csv')
test = TabularDataset('data/input/titanic/test.csv')

# Added the following 6 lines of code
train_df = pd.read_csv('data/input/titanic/train.csv')
test_df = pd.read_csv('data/input/titanic/test.csv')
train_columns = train_df.columns.to_list()
test_columns = test_df.columns.to_list()
tmp = list(set(train_columns) - set(test_columns))
label = tmp[0]

# No need for the following code
# label='Survived'
time_limit=60