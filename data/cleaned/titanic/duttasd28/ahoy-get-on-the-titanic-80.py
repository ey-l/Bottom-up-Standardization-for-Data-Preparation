#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython import display
display.Image('https://raw.githubusercontent.com/Dutta-SD/Images_Unsplash/master/Kaggle/dorian-mongel-5Rgr_zI7pBw-unsplash.jpg', width = 3000, height = 500)


# # Titanic - An Introduction to Neural Networks
# 
# The titanic disaster is one of the major disasters that the world has faced. It led to tragic loss of lives and destruction of the beautiful Titanic ship.
# 
# # Objective-To predict survival with simple explanation
# 
# Let us use Machine Learning to try to predict which passengers survived and which passengers did not. I will try to explain as simply as possible so that beginners can unserstand it easily

# # Importing the data
# * In Machine Learning, our objective is to train a model which will 'train' from some data and then predict on 'test' data.
# * So, first we would read data using pandas.
# * Here, the train data and the test data are named _train.csv_ and _test.csv_

# In[ ]:


# Import Necessary libraries
import pandas as pd
import numpy as np


# In[ ]:


# import dataset
train_data = pd.read_csv('data/input/titanic/train.csv')
test_data = pd.read_csv('data/input/titanic/test.csv')


# In[ ]:


# Head of training data
train_data.head()


# In[ ]:


# Head of submission file
test_data.head()


# * The Name, Ticket, Cabin, PassengerId columns do not seem to be meaningful. Let us drop those columns.
# 
# * The Cabin column is full of _NaN_ values, that is null values, so we should better drop it.
# 
# * Dropping Data might lead to loss of accuracy

# In[ ]:


PassengerID = test_data.PassengerId
## code for dropping data
train_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True, axis=1)
test_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True, axis=1)

test_data.head()


# # Null Value Management
# Now, let us try and see if there are null values or not. Null or NAN values represent data that is missing.
# Dropping them might be useful sometimes, but it is generally better to replace them with some suitable value.

# In[ ]:


# Check for NaN values
print(train_data.isnull().any())


# In[ ]:


test_data.isnull().any()


# So we see that there are indeed null values.We are going to fix them real soon.

# # Split into independent and dependent features
# 
# * We are trying to predict whether the passenger Survived or not. 
# * So let us take the feature we want to predict to be 'y' and the training data to be X.
# 
# We will use X to predict y. So X is called Independent features and y is called dependent feautures. 

# In[ ]:


# Split into dependent and independent dataframes
y = train_data.Survived

# drop the Survived columns from the independent features
## Retaining only dependendt features
X = train_data.drop(['Survived'], axis = 1)

print(y.head())
print(X.head())


# # Exploratory Data Analysis
# * Visualisation will help us explore the data. This can give us a lot of important information about the nature of data.
# * This step should never be skipped.

# In[ ]:


## Useful Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns


# ### 1. BarPlot 
# This plot is used to visualise the number of passengers that survived v/s passengers that died

# In[ ]:


# survived
sns.barplot(x = y.unique(), y = y.value_counts());


# ### 2. PairPlot
# Let us visualise the pairplot between different varibles to see their relation to each other

# In[ ]:


# Pairplot
sns.pairplot(data = train_data, corner = True, palette = 'summer');


# # Training Phase
# 1. Now we start to train the model
# 2. We first split the data to train and validation set.
# 3. Validation set is necessary so that you have some estimate on how the model performs on unseen data.
# 4. Later you train the model on the whole data so that its performance increases

# In[ ]:


X_train, X_test, y_train, y_test = X, test_data, y, None


# In[ ]:


# The indexes are random order, we need to reset them
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

X_train.info()


# In[ ]:


# Lets separate the object data from numerical data
s = (X_train.dtypes=='object')
categorical_cols = list(s[s].index)

# Get numerical data column names
numerical_cols = [ i for i in X_train.columns if not i in categorical_cols ]
numerical_cols


# # Filling NULL Values with KNN
# 
# 1. KNN Imputer imputes(fills null values) by using KNN. 
# 2. It takes k nearest data points of the point with missing values and fill the missing value in.

# In[ ]:


from sklearn.impute import KNNImputer
##from sklearn.preprocessing import StandardScaler   ## We turned off scaling here, you can try if you want

# Imputer Object
nm_imputer = KNNImputer()
## ss is the scaler, you can try it if you want
### We will not scale here
# ss = StandardScaler()

# Transform the necessary columns
X_train_numerical = pd.DataFrame(nm_imputer.fit_transform(X_train[numerical_cols]),
                                 columns = numerical_cols)
###X_train_numerical = pd.DataFrame(ss.fit_transform(X_train_numerical[numerical_cols]), columns = numerical_cols)

X_test_numerical = pd.DataFrame(nm_imputer.transform(X_test[numerical_cols]),
                                 columns = numerical_cols)
#X_test_numerical = pd.DataFrame(ss.transform(X_test_numerical[numerical_cols]), columns = numerical_cols)


# In[ ]:


# Drop the non required columns(with missing values)
X_train = X_train.drop(numerical_cols, axis = 1)
X_test = X_test.drop(numerical_cols, axis = 1)

# put new colums in dataframe by joining
X_train = X_train.join(X_train_numerical)
X_test = X_test.join(X_test_numerical)

X_train.isnull().any()


# # Simple Imputer
# Simple imputer imputes values with the values it is provided

# In[ ]:


# Impute categorical columns
from sklearn.impute import SimpleImputer

# Imputer Object
nm_imputer = SimpleImputer(strategy='most_frequent')

# Transform the necessary columns
X_train_numerical = pd.DataFrame(nm_imputer.fit_transform(X_train[categorical_cols]),
                                 columns = categorical_cols)

X_test_numerical = pd.DataFrame(nm_imputer.transform(X_test[categorical_cols]),
                                 columns = categorical_cols)


# In[ ]:


# Drop the non required columns(with missing values)
X_train = X_train.drop(categorical_cols, axis = 1)
X_test = X_test.drop(categorical_cols, axis = 1)

# put new colums in dataframe
X_train = X_train.join(X_train_numerical)
X_test = X_test.join(X_test_numerical)

X_train.isnull().any()


# # One Hot Encoder
# 
# 1. Let us say we have some data like male or female.
# 2. We can then create a column like isFemale where if it is 0, it would denote female else male
# 3. This is the idea behind One hot encoding.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[categorical_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[categorical_cols]) )

#Reset the index
OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index

# Remove Categorical Columns
num_X_train = X_train.drop(categorical_cols, axis = 1)
num_X_test = X_test.drop(categorical_cols, axis = 1)

# Join
X_train = num_X_train.join(OH_cols_train, how='left')
X_test = num_X_test.join(OH_cols_test, how='left')

X_train.head()

                             


# In[ ]:


X_test.info()


# In[ ]:


X_train.info()


# # Neural Network
# 
# Finally, we will create our neural network model.
# 
# A neural network looks like this:
# ![Neural Network image](https://raw.githubusercontent.com/Dutta-SD/Images_Unsplash/master/Kaggle/Screenshot%20from%202020-08-24%2012-26-52.png)
# **Image taken from dair.ai github repository**
# * The inputs are given in the input layer, it passes through the hidden layers which transforms it. 
# * It add's weights to the inputs and adds a term called bias. This helps to figure out which figure are important
# * It also uses some special functions called 'activations' which helps in giving non linearity to the network

# In[ ]:


# Create a validation set 
from sklearn.model_selection import train_test_split

X_train_2, X_val, y_train_2,  y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 10)


# # And Finally, KERAS!
# Keras is the library we will be using for

# In[ ]:


from tensorflow import keras


# In[ ]:


from keras import Sequential
from keras.layers import BatchNormalization, Dense
## Dropout is a form of regularisation for neural networks


# # 1. Create Model

# In[ ]:


model = Sequential()

model.add(Dense(128, activation = 'relu', input_shape = (10,) ))
model.add(BatchNormalization())
model.add(Dense(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(8, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation = 'sigmoid'))

model.summary()


# # 2. Compile It

# In[ ]:


model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy'])


# # Initial Test to see how well the model is performing.
# After we are done, we will train using the entire dataset

# In[ ]:

