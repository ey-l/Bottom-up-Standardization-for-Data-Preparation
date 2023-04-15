import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
d_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
d_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
d_train.info()
print('First 5 raws of data:')
d_train.head()
print('Last 5 raws of data:')
d_train.tail()
d_train.describe().T
categorical_features = []
threshold = 20
for each in d_train.columns:
    if d_train[each].nunique() < threshold:
        categorical_features.append(each)
numerical_features = []
for each in d_train.columns:
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
    sns.countplot(data=d_train, x=var, alpha=0.3, color='red')
    sns.countplot(data=d_test, x=var, alpha=0.5, color='green')
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(30, 40))
for (i, var) in enumerate(numerical_features[1:]):
    if var == 'SalePrice':
        break
    else:
        plt.subplot(6, 4, i + 1)
        plt.hist(d_train[var], bins=50, color='red', alpha=0.5, label='Train Data')
        plt.xlabel(var)
        plt.ylabel('Count')
        plt.legend()
        plt.hist(d_test[var], bins=50, color='green', alpha=0.5, label='Test Data')
        plt.legend()
outlier_indexes = []

def outlier_plotting(feature):
    outlier = []
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=d_train[feature], palette='Set3')
    plt.title("{}'s Outlier Box Plot".format(feature), weight='bold')
    plt.xlabel(feature, weight='bold')

    Q1 = d_train[feature].quantile(0.25)
    Q3 = d_train[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_outlier_limit = Q1 - 1.5 * IQR
    higher_outlier_limit = Q3 + 1.5 * IQR
    print('Values lower than {} and higher than {} are outliers for {}.\n'.format(lower_outlier_limit, higher_outlier_limit, feature))
    threshold = 3
    for i in d_train[feature]:
        z = (i - d_train[feature].mean()) / d_train[feature].std()
        if z > threshold:
            outlier.append(i)
            index = d_train[d_train[feature] == i].index[0]
            outlier_indexes.append(index)
    if outlier == []:
        print('No any outliers for {}.'.format(feature))
    else:
        print('There are {} outliers for {}:'.format(len(outlier), feature), outlier)
for i in [col for col in d_train.columns if d_train[col].dtype != 'O']:
    if i != 'Id':
        outlier_plotting(i)
d_train.loc[outlier_indexes]
d_train = d_train.drop(outlier_indexes, axis=0).reset_index(drop=True)
d_train.info()
alldata = pd.concat([d_train, d_test], axis=0, sort=False)
alldata['SalePrice'].head()
alldata['SalePrice'].tail()
pd.set_option('display.max_rows', 100)
info_count = pd.DataFrame(alldata.isnull().sum(), columns=['Count of NaN'])
dtype = pd.DataFrame(alldata.dtypes, columns=['DataTypes'])
info = pd.concat([info_count, dtype], axis=1)
info
from sklearn.preprocessing import LabelEncoder
alldata['LotFrontage'].interpolate(method='linear', inplace=True)
for i in info.T:
    if i == 'Id' or i == 'SalePrice' or i == 'LotFrontage':
        continue
    elif info.T[i][0] == 0:
        continue
    elif info.T[i][0] < 400:
        alldata[i].fillna(alldata[i].value_counts().index[0], inplace=True)
    else:
        lbl_enc = LabelEncoder()