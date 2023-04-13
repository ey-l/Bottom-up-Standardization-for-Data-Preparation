import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head(10)
_input1.info()
_input1.shape
_input1.describe()
for feature in _input1.columns:
    percentage_of_null_values = _input1[feature].isnull().sum() / len(_input1[feature]) * 100
    print(f'There is a total number of {percentage_of_null_values} % null values in {feature}')
from sklearn.impute import SimpleImputer
for feature in _input1.columns:
    if _input1[feature].dtypes == 'object':
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')