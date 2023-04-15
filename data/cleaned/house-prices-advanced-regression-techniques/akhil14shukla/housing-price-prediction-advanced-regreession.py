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
df = pd.read_csv('https://raw.githubusercontent.com/akhil14shukla/Housing-Price-Prediction/master/train.csv', index_col='Id')
df_test = pd.read_csv('https://raw.githubusercontent.com/akhil14shukla/Housing-Price-Prediction/master/test.csv', index_col='Id')
sns.catplot(x='YearRemodAdd', y='SalePrice', data=df, aspect=2)
df.value_counts()
sns.catplot(x='GarageCars', y='SalePrice', data=df)
sns.lineplot(x='GarageCars', y='SalePrice', data=df)
sns.lineplot(x='GarageCars', y='GarageArea', data=df)
from sklearn.linear_model import LinearRegression
sub = df[['OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'EnclosedPorch']]