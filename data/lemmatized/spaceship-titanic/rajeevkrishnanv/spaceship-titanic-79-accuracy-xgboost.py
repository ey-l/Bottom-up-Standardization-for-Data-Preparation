import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input1.describe(include='bool')
_input1.info()
_input1.describe()
_input1.isnull().sum() / _input1.count() * 100

def get_group_id(passengerId: str):
    return passengerId.split('_')[0]
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocessing(df_train):
    _input1['Group'] = _input1['PassengerId'].map(get_group_id)
    _input1['GroupSize'] = _input1.groupby('Group')['Group'].transform('count')
    column_names = _input1.columns
    imputer = SimpleImputer(strategy='most_frequent')