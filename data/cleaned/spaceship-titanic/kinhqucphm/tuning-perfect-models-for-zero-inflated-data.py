import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_train.head(10)
df_train.info()
df_train.shape
df_train.describe()
for feature in df_train.columns:
    percentage_of_null_values = df_train[feature].isnull().sum() / len(df_train[feature]) * 100
    print(f'There is a total number of {percentage_of_null_values} % null values in {feature}')
from sklearn.impute import SimpleImputer
for feature in df_train.columns:
    if df_train[feature].dtypes == 'object':
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')