import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
filepath_train = '_data/input/house-prices-advanced-regression-techniques/train.csv'
train_data = pd.read_csv(filepath_train)
train_data.head()
train_data.describe()
train_data.info()
columns = [col for col in train_data.columns if not train_data[col].dtype == 'object']
train_data[columns]
corr_mat = train_data.corr()
corr_mat['SalePrice'].sort_values(ascending=False)
columns.remove('Id')
new_data = train_data[columns]
cols_with_missing = [col for col in new_data.columns if train_data[col].isnull().any() and (not train_data[col].dtype == 'object')]
print(cols_with_missing)
new_data.head()
new_data.drop('SalePrice', axis=1)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')