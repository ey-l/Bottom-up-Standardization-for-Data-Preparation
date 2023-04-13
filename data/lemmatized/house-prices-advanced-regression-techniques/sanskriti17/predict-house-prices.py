import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.head()
_input1.shape
_input1.info()
train_data_numeric = _input1.select_dtypes(include='number')
print(train_data_numeric)
train_data_numeric.describe()
import seaborn as sns
import matplotlib.pyplot as plt
corrmat = train_data_numeric.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(30, 30))
g = sns.heatmap(train_data_numeric[top_corr_features].corr(), annot=True, cmap='RdYlGn')
print(train_data_numeric.columns)
X_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
df = train_data_numeric[X_features]
df.head(6)
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
df_drop = train_data_numeric.dropna()
X = df_drop.iloc[:, 0:37]
y = df_drop.iloc[:, -1]
model = ExtraTreesClassifier()