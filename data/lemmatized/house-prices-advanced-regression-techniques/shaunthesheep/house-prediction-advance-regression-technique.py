import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost
sns.set(style='white')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.tail()
_input1.shape
y = _input1[['Id', 'SalePrice']]
_input1 = _input1.drop('SalePrice', axis=1)
final_set = pd.concat([_input1, _input0], axis=0)
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, final_set.shape[1])), columns=final_set.columns)
corr = d.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
(f, ax) = plt.subplots(figsize=(30, 20))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})
final_set.tail()
print('Total Features :: ', len(final_set.columns.tolist()))
print('Total Categorical variables::', len(final_set.select_dtypes(exclude=['number', 'bool_']).columns.tolist()))
print('Total Continuous variables :: ', len(final_set.select_dtypes(exclude=['object_']).columns.tolist()))
(fig, ax) = plt.subplots(figsize=(30, 15))
sns.heatmap(final_set.isnull(), yticklabels=False, cbar=False, ax=ax)
final_set = final_set.loc[:, final_set.isnull().mean() < 0.4]
(fig, ax) = plt.subplots(figsize=(30, 15))
sns.heatmap(final_set.isnull(), yticklabels=False, cbar=False, ax=ax)
categorical_features = final_set.select_dtypes(exclude=['number', 'bool_']).columns.tolist()
len(categorical_features)

def fill_nan_categorical(categorical_features, final_set):
    for i in categorical_features:
        final_set[i] = final_set[i].fillna(final_set[i].mode()[0])
    return final_set
final_set = fill_nan_categorical(categorical_features, final_set)
(fig, ax) = plt.subplots(figsize=(40, 20))
sns.heatmap(final_set.isnull(), yticklabels=False, cbar=False, ax=ax)
continuous_features = final_set.select_dtypes(exclude=['object_']).columns.tolist()

def fill_nan_continuous(continuous_features, final_set):
    for i in continuous_features:
        final_set[i] = final_set[i].fillna(final_set[i].mean())
    return final_set
final_set = fill_nan_continuous(continuous_features, final_set)
(fig, ax) = plt.subplots(figsize=(40, 20))
sns.heatmap(final_set.isnull(), yticklabels=False, cbar=False)

def one_hot_encoder(final_set):
    df = final_set.copy(deep=True)
    dummies = pd.get_dummies(df, prefix='column_', drop_first=True)
    return dummies
final_set = one_hot_encoder(final_set)
final_set = final_set.loc[:, ~final_set.columns.duplicated()]
final_set.head()
train_data = pd.DataFrame(final_set[:1460])
test_data = pd.DataFrame(final_set[1460:2920])
print(train_data.shape)
print(test_data.shape)
X = final_set[:1460]
(X_train, X_test, y_train, y_test) = train_test_split(X, y['SalePrice'], test_size=0.1, random_state=42)
xgb_classifier = xgboost.XGBRFRegressor(n_estimators=20, learning_rate=1, reg_lambda=0.1, gamma=0.6, max_depth=10)