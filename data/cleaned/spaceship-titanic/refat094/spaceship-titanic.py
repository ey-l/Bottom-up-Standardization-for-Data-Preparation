import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
titanic_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
titanic_df
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_df
titanic_df.info()
test_df.info()
missing_counts = titanic_df.isna().sum().sort_values(ascending=False)
missing_counts
titanic_df.nunique().sort_values(ascending=False)
titanic_df.describe()
input_df = titanic_df
input_df[['Cabin_part1', 'Cabin_part2', 'Cabin_part3']] = input_df['Cabin'].str.split('/', expand=True).astype(str)
input_df
input_df = input_df.drop('Cabin', axis=1)
input_df = input_df.drop('Cabin_part2', axis=1)
input_df = input_df.drop('Name', axis=1)
input_df
test_df[['Cabin_part1', 'Cabin_part2', 'Cabin_part3']] = test_df['Cabin'].str.split('/', expand=True).astype(str)
test_df
test_df = test_df.drop('Cabin', axis=1)
test_df = test_df.drop('Cabin_part2', axis=1)
test_df = test_df.drop('Name', axis=1)
test_df
target_col = input_df.columns[-3]
target_col
input_cols = input_df.columns[1:]
input_cols = input_cols.drop(['Transported'])
input_cols
(input_df, targets) = (titanic_df[input_cols].copy(), titanic_df[target_col].copy())
test_df = test_df[input_cols].copy()
numeric_cols = input_df[input_cols].select_dtypes(include=np.number).columns.tolist()
categorical_cols = input_df[input_cols].select_dtypes(exclude=np.number).columns.tolist()
categorical_cols
numeric_cols
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split