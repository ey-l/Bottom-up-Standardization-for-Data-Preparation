import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col=0)
_input1.head()
_input1['SalePrice'].describe()
sns.distplot(_input1['SalePrice'])
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(_input1[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(_input1[cols], size=2.5)
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
_input1 = _input1.fillna(_input1.mean())
saleprice_scaled = StandardScaler().fit_transform(_input1['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
var = 'GrLivArea'
data = pd.concat([_input1['SalePrice'], _input1[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']
_input1 = _input1[cols]
_input1 = pd.get_dummies(_input1)
_input1 = _input1.fillna(_input1.mean())
scale = StandardScaler()
X_train = _input1[['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']]
X_train = scale.fit_transform(X_train)
y = _input1['SalePrice'].values
seed = 7
np.random.seed(seed)
(X_train, X_test, y_train, y_test) = train_test_split(X_train, y, test_size=0.33, random_state=seed)

def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metrics.mae])
    return model
model = create_model()
model.summary()