#!/usr/bin/env python
# coding: utf-8

# # Import Packages
# Lets load all the needed packages for this notebook:

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns


# In[ ]:


import tensorflow as tf
tf.__version__


# # The Dataset
# For this notebook we will use the Titanic competition dataset.
# 
# Let's define the path to the dataset:

# In[ ]:


train_csv_path: str = 'data/input/titanic/train.csv'

data: pd.DataFrame = pd.read_csv(train_csv_path)
data.info()


# # Quick Look at the Data
# Let’s take a look at the top five rows:

# In[ ]:


data.head()


# # Remove any columns that aren't needed from the dataset.

# In[ ]:


data = data.drop(['PassengerId','Name','Ticket','Cabin','Parch'],axis=1)
data.head()


# # Checking null values

# In[ ]:


data.isnull().sum()


# In[ ]:


data.dropna(subset=['Embarked'], inplace=True)


# # Filling null values

# In[ ]:


data['Age'].fillna(data['Age'].mean(),inplace = True)


# In[ ]:


data.isnull().sum()


# In[ ]:


sex_col = data['Sex'] == 'male'
sex_col = sex_col.astype('int32')


data = data.drop(['Sex'],axis=1)

data['Sex'] = sex_col

data.head()


# In[ ]:


data = pd.get_dummies(data, columns = ['Embarked'])
data.head()


# # Split Data

# In[ ]:


X = data.drop('Survived', axis=1).to_numpy()
y = data['Survived'].to_numpy()


# In[ ]:


X.shape, y.shape


# # Feature scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# # Splitting traning set

# In[ ]:


from sklearn.model_selection import train_test_split

tf.random.set_seed(42)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# # Building and Training our model

# In[ ]:


# let's build a model to find patterns in it

# Set random seed
tf.random.set_seed(42)

# 1. Create a model
model_1 = tf.keras.Sequential([
           tf.keras.layers.Dense(9, activation='relu'),
           tf.keras.layers.Dense(15, activation='relu'),
           tf.keras.layers.Dense(50, activation='relu'),
           tf.keras.layers.Dense(2, activation='softmax')
])

# 2. Comile the model
model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 metrics=['accuracy'])

# 3. Fit the model