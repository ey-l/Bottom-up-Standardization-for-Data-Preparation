import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Train data shape:', _input1.shape)
print('Test data shape:', _input0.shape)
_input1.head()
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
_input1.SalePrice.describe()
print('Skew is:', _input1.SalePrice.skew())
plt.hist(_input1.SalePrice, color='blue')
target = np.log(_input1.SalePrice)
print('Skew is:', target.skew())
plt.hist(target, color='blue')
numeric_features = _input1.select_dtypes(include=[np.number])
numeric_features.dtypes
corr = numeric_features.corr()
print(corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print(corr['SalePrice'].sort_values(ascending=False)[-5:])
_input1.OverallQual.unique()
quality_pivot = _input1.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
quality_pivot
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.scatter(x=_input1['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.scatter(x=_input1['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
_input1 = _input1[_input1['GarageArea'] < 1200]
plt.scatter(x=_input1['GarageArea'], y=np.log(_input1.SalePrice))
plt.xlim(-200, 1600)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
nulls = pd.DataFrame(_input1.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls
print('Unique values are:', _input1.MiscFeature.unique())
categoricals = _input1.select_dtypes(exclude=[np.number])
categoricals.describe()
print('Original: \n')
print(_input1.Street.value_counts(), '\n')
_input1['enc_street'] = pd.get_dummies(_input1.Street, drop_first=True)
_input0['enc_street'] = pd.get_dummies(_input1.Street, drop_first=True)
print('Encoded: \n')
print(_input1.enc_street.value_counts())
condition_pivot = _input1.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)

def encode(x):
    return 1 if x == 'Partial' else 0
_input1['enc_condition'] = _input1.SaleCondition.apply(encode)
_input0['enc_condition'] = _input0.SaleCondition.apply(encode)
condition_pivot = _input1.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
data = _input1.select_dtypes(include=[np.number]).interpolate().dropna()
sum(data.isnull().sum() != 0)
X = data.drop(['SalePrice', 'Id'], axis=1)
y = np.log(_input1.SalePrice)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=42, test_size=0.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()