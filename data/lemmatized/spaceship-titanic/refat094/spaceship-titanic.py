import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0
_input1.info()
_input0.info()
missing_counts = _input1.isna().sum().sort_values(ascending=False)
missing_counts
_input1.nunique().sort_values(ascending=False)
_input1.describe()
input_df = _input1
input_df[['Cabin_part1', 'Cabin_part2', 'Cabin_part3']] = input_df['Cabin'].str.split('/', expand=True).astype(str)
input_df
input_df = input_df.drop('Cabin', axis=1)
input_df = input_df.drop('Cabin_part2', axis=1)
input_df = input_df.drop('Name', axis=1)
input_df
_input0[['Cabin_part1', 'Cabin_part2', 'Cabin_part3']] = _input0['Cabin'].str.split('/', expand=True).astype(str)
_input0
_input0 = _input0.drop('Cabin', axis=1)
_input0 = _input0.drop('Cabin_part2', axis=1)
_input0 = _input0.drop('Name', axis=1)
_input0
target_col = input_df.columns[-3]
target_col
input_cols = input_df.columns[1:]
input_cols = input_cols.drop(['Transported'])
input_cols
(input_df, targets) = (_input1[input_cols].copy(), _input1[target_col].copy())
_input0 = _input0[input_cols].copy()
numeric_cols = input_df[input_cols].select_dtypes(include=np.number).columns.tolist()
categorical_cols = input_df[input_cols].select_dtypes(exclude=np.number).columns.tolist()
categorical_cols
numeric_cols
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split