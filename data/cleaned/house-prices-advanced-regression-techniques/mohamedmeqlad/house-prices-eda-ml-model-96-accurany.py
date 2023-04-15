import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm, skew
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=14, titlepad=10)
sns.set_theme(style='darkgrid')
plt.rcParams.update({'figure.figsize': (15, 10)})
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
train.describe().T.style.set_properties(**{'background-color': 'grey', 'color': 'white', 'border-color': 'white'})
test.describe().T.style.set_properties(**{'background-color': 'grey', 'color': 'white', 'border-color': 'white'})
fig_ = train.hist(figsize=(25, 30), bins=50, xlabelsize=8, ylabelsize=8)
fig_ = test.hist(figsize=(25, 30), bins=50, color='red', edgecolor='black', xlabelsize=8, ylabelsize=8)
(fig, axes) = plt.subplots(1, 2, sharex=True, figsize=(20, 10))
sns.heatmap(ax=axes[0], yticklabels=False, data=train.isnull(), cbar=False)
sns.heatmap(ax=axes[1], yticklabels=False, data=test.isnull(), cbar=False, cmap='YlGnBu')
axes[0].set_title('Heatmap of missing values in training data')
axes[1].set_title('Heatmap of missing values in testing data')

plt.style.use('ggplot')
missing_values = train.isna().sum(axis=0) / train.shape[0]
missing_values = missing_values.loc[missing_values > 0]
missing_values = missing_values.sort_values(ascending=False)
missing_values.plot(kind='bar', title='Missing Columns', ylabel='% of missing', ylim=(0, 1.2))
y_train = train['SalePrice']
data = pd.concat([train, test], axis=0)
data = data.drop(['Id', 'SalePrice'], axis=1)
number_of_missing = data.isnull().sum().sort_values()
percent_of_missing = (data.isnull().sum() / data.isnull().count() * 100).sort_values()
missing = pd.concat([number_of_missing, percent_of_missing], keys=['total number of missing data', 'total percent of missing data'], axis=1)
print(missing.tail(10))
data = data.drop(missing[missing['total number of missing data'] > 5].index, axis=1)
data.isnull().sum().sort_values(ascending=False)
categoric = ['MSZoning', 'LotShape', 'LotConfig', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'ExterQual', 'Foundation', 'HeatingQC', 'Electrical', 'KitchenQual', 'SaleCondition']
numeric_data = [column for column in data.select_dtypes(['int', 'float'])]
categoric_data = data[categoric]
for col in categoric_data:
    data[col].fillna(data[col].value_counts().index[0], inplace=True)
x = data[numeric_data]
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
data[numeric_data] = pd.DataFrame(imputer.fit_transform(x))
data[numeric_data].isna().sum().sort_values(ascending=False)
print('Number of missing data is', data.isna().sum().sum())
numeric_data = [column for column in data.select_dtypes(['number'])]
vars_skewed = data[numeric_data].apply(lambda x: skew(x)).sort_values()
vars_skewed
for var in vars_skewed.index:
    data[var] = np.log1p(data[var])
data = pd.get_dummies(data, drop_first=True)
data.head()
X_train = data[:len(train)]
X_test = data[len(train):]
(X_train.shape, X_test.shape)
reg = GradientBoostingRegressor(random_state=42, loss='ls', learning_rate=0.1)