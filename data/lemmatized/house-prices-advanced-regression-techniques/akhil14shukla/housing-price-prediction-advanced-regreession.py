import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
_input1 = pd.read_csv('https://raw.githubusercontent.com/akhil14shukla/Housing-Price-Prediction/master/train.csv', index_col='Id')
_input0 = pd.read_csv('https://raw.githubusercontent.com/akhil14shukla/Housing-Price-Prediction/master/test.csv', index_col='Id')
sns.catplot(x='YearRemodAdd', y='SalePrice', data=_input1, aspect=2)
_input1.value_counts()
sns.catplot(x='GarageCars', y='SalePrice', data=_input1)
sns.lineplot(x='GarageCars', y='SalePrice', data=_input1)
sns.lineplot(x='GarageCars', y='GarageArea', data=_input1)
from sklearn.linear_model import LinearRegression
sub = _input1[['OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'EnclosedPorch']]