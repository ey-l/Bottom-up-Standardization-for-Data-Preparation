import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.info()
print('First 5 raws of data:')
_input1.head()
print('Last 5 raws of data:')
_input1.tail()
_input1.describe().T
categorical_features = []
threshold = 20
for each in _input1.columns:
    if _input1[each].nunique() < threshold:
        categorical_features.append(each)
numerical_features = []
for each in _input1.columns:
    if each not in categorical_features:
        numerical_features.append(each)
print('Categorical Variables:\n\n', categorical_features, '\n\n')
print('Numerical Variables:\n\n', numerical_features)
numerical_features
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(30, 70))
for (i, var) in enumerate(categorical_features):
    plt.subplot(15, 4, i + 1)
    sns.countplot(data=_input1, x=var, alpha=0.3, color='red')
    sns.countplot(data=_input0, x=var, alpha=0.5, color='green')
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(30, 40))
for (i, var) in enumerate(numerical_features[1:]):
    if var == 'SalePrice':
        break
    else:
        plt.subplot(6, 4, i + 1)
        plt.hist(_input1[var], bins=50, color='red', alpha=0.5, label='Train Data')
        plt.xlabel(var)
        plt.ylabel('Count')
        plt.legend()
        plt.hist(_input0[var], bins=50, color='green', alpha=0.5, label='Test Data')
        plt.legend()
outlier_indexes = []

def outlier_plotting(feature):
    outlier = []
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=_input1[feature], palette='Set3')
    plt.title("{}'s Outlier Box Plot".format(feature), weight='bold')
    plt.xlabel(feature, weight='bold')
    Q1 = _input1[feature].quantile(0.25)
    Q3 = _input1[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_outlier_limit = Q1 - 1.5 * IQR
    higher_outlier_limit = Q3 + 1.5 * IQR
    print('Values lower than {} and higher than {} are outliers for {}.\n'.format(lower_outlier_limit, higher_outlier_limit, feature))
    threshold = 3
    for i in _input1[feature]:
        z = (i - _input1[feature].mean()) / _input1[feature].std()
        if z > threshold:
            outlier.append(i)
            index = _input1[_input1[feature] == i].index[0]
            outlier_indexes.append(index)
    if outlier == []:
        print('No any outliers for {}.'.format(feature))
    else:
        print('There are {} outliers for {}:'.format(len(outlier), feature), outlier)
for i in [col for col in _input1.columns if _input1[col].dtype != 'O']:
    if i != 'Id':
        outlier_plotting(i)
_input1.loc[outlier_indexes]
_input1 = _input1.drop(outlier_indexes, axis=0).reset_index(drop=True)
_input1.info()
alldata = pd.concat([_input1, _input0], axis=0, sort=False)
alldata['SalePrice'].head()
alldata['SalePrice'].tail()
pd.set_option('display.max_rows', 100)
info_count = pd.DataFrame(alldata.isnull().sum(), columns=['Count of NaN'])
dtype = pd.DataFrame(alldata.dtypes, columns=['DataTypes'])
info = pd.concat([info_count, dtype], axis=1)
info
from sklearn.preprocessing import LabelEncoder
alldata['LotFrontage'] = alldata['LotFrontage'].interpolate(method='linear', inplace=False)
for i in info.T:
    if i == 'Id' or i == 'SalePrice' or i == 'LotFrontage':
        continue
    elif info.T[i][0] == 0:
        continue
    elif info.T[i][0] < 400:
        alldata[i] = alldata[i].fillna(alldata[i].value_counts().index[0], inplace=False)
    else:
        lbl_enc = LabelEncoder()