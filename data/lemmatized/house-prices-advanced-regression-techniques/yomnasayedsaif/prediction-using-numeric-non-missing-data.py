import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.columns
test_id = _input0['Id']
missing = _input1.isna()
percent = (missing.sum() / missing.count() * 100).sort_values(ascending=False)
missing_columns = percent[percent > 0].index.tolist()
_input1 = _input1.drop(missing_columns, axis=1, inplace=False)
X = _input1[['LotArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MSSubClass', 'OverallQual', 'OverallCond', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces']]
y = _input1['SalePrice']
import seaborn as sns
sns.distplot(y)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2, random_state=1)
y_train_l = np.log1p(y_train)
y_val_l = np.log1p(y_val)
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_val = s_scaler.transform(X_val.astype(np.float))
regressor = LinearRegression()