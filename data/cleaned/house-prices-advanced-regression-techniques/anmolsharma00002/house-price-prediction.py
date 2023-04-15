import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew
from xgboost import XGBRegressor
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train.head()
df_test.head()
df_train.info()
df_train['SalePrice'].describe()
sns.histplot(df_train['SalePrice'])
sns.histplot(np.log1p(df_train['SalePrice']))
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
corr = df_train.corr()
corr['SalePrice'].sort_values(ascending=False)
y = df_train['SalePrice']
test_id = df_test['Id']
all_df = pd.concat([df_train, df_test], axis=0, sort=False)
all_df.drop(['Id', 'SalePrice'], axis=1)
Total = all_df.isnull().sum().sort_values(ascending=False)
missing_data = pd.concat([Total], axis=1, keys=['Total'])
missing_data.head(30)
all_df.drop(missing_data[missing_data['Total'] > 5].index, axis=1, inplace=True)
all_df.isnull().sum().max()
total = all_df.isnull().sum().sort_values(ascending=False)
total.head(30)
numeric_missed = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageArea', 'GarageCars']
for feature in numeric_missed:
    all_df[feature] = all_df[feature].fillna(0)
categorical_missed = ['Exterior1st', 'Exterior2nd', 'SaleType', 'MSZoning', 'Electrical', 'KitchenQual']
for feature in categorical_missed:
    all_df[feature] = all_df[feature].fillna(all_df[feature].mode()[0])
all_df['Functional'] = all_df['Functional'].fillna('Typ')
all_df.drop(['Utilities'], axis=1, inplace=True)
all_df.isnull().sum().max()
all_df = pd.get_dummies(all_df)
all_df
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
all_scaled = pd.DataFrame(Scaler.fit_transform(all_df))
train_cleaned = pd.DataFrame(all_scaled[:1460])
test_cleaned = pd.DataFrame(all_scaled[1460:])
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(train_cleaned, y, test_size=0.3, random_state=20)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
reg = LinearRegression()