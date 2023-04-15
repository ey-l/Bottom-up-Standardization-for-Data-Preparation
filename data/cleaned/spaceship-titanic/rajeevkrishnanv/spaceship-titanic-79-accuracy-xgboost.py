import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.head()
df_train.describe(include='bool')
df_train.info()
df_train.describe()
df_train.isnull().sum() / df_train.count() * 100

def get_group_id(passengerId: str):
    return passengerId.split('_')[0]
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocessing(df_train):
    df_train['Group'] = df_train['PassengerId'].map(get_group_id)
    df_train['GroupSize'] = df_train.groupby('Group')['Group'].transform('count')
    column_names = df_train.columns
    imputer = SimpleImputer(strategy='most_frequent')