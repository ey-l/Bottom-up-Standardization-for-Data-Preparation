import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data
data.head()
data.tail()
data.describe()
data.info()
data.duplicated().sum()
pd.options.display.max_rows = None
data.isnull().sum()
pd.reset_option('max_rows')
data.LotFrontage.value_counts()
sns.distplot(data.LotFrontage)

data['LotFrontage'].mean()
print('The Percentage of data missing in LotFrontage is ', data.LotFrontage.isnull().sum() / len(data) * 100)
data.loc[data['LotFrontage'].isnull() == True]
data.loc[data['LotFrontage'].isnull() == True, 'LotFrontage'] = 70.0
d1 = data[['Alley', 'PoolQC', 'Fence', 'MiscFeature']]
a = d1.isnull().sum() / len(d1) * 100
a
data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
d2 = data[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']]
b = d2.isnull().sum() / len(d2) * 100
b
data['BsmtQual'] = data['BsmtQual'].fillna('NA')
data['BsmtCond'] = data['BsmtCond'].fillna('NA')
data['BsmtExposure'] = data['BsmtExposure'].fillna('NA')
data['BsmtFinType1'] = data['BsmtFinType1'].fillna('NA')
data['BsmtFinType2'] = data['BsmtFinType2'].fillna('NA')
data.MasVnrType.value_counts()
data.loc[data['MasVnrType'].isnull() == True, 'MasVnrType'] = 'None'
sns.distplot(data.MasVnrArea)
data['MasVnrArea'].median()
data.loc[data['MasVnrArea'].isnull() == True, 'MasVnrArea'] = 0.0
d3 = data[['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']]
c = d3.isnull().sum() / len(d3) * 100
c
data['GarageType'] = data['GarageType'].fillna('NA')
data['GarageFinish'] = data['GarageFinish'].fillna('NA')
data['GarageQual'] = data['GarageQual'].fillna('NA')
data['GarageCond'] = data['GarageCond'].fillna('NA')
sns.distplot(data.GarageYrBlt)
data.GarageYrBlt.median()
data.loc[data['GarageYrBlt'].isnull() == True, 'GarageYrBlt'] = 1980.0
data.Electrical.value_counts()
data.loc[data['Electrical'].isnull() == True, 'Electrical'] = 'SBrkr'
data.FireplaceQu.value_counts()
data.loc[data['FireplaceQu'].isnull() == True, 'FireplaceQu'] = 'NA'
data.isnull().sum()
plt.figure(figsize=(25, 25))
sns.heatmap(data.drop('SalePrice', axis=1).corr(), cmap='BuPu', annot=True)

dtype_objects = list((columns for columns in data.select_dtypes([object]).columns))
dtype_objects
len(dtype_objects)
plt.figure(figsize=(15, 150))
plotnumber = 1
for c in dtype_objects:
    ax = plt.subplot(20, 2, plotnumber)
    b = sns.countplot(x=data[c], palette='Set2')
    plt.xticks(rotation=70)
    plotnumber += 1
    for bar in b.patches:
        b.annotate(format(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=10, xytext=(0, 6), textcoords='offset points')

dtype_float = list((columns for columns in data.select_dtypes([float]).columns))
dtype_float
plt.figure(figsize=(15, 150))
plotnumber = 1
for a in dtype_float:
    ax = plt.subplot(20, 2, plotnumber)
    sns.distplot(x=data[a], color='purple')
    plt.xticks(rotation=70)
    plotnumber += 1

plt.figure(figsize=(5, 5))
labels = ['Y', 'N']
size = data['CentralAir'].value_counts()
colors = ['lightgreen', 'lightslategray']
explode = [0, 0.3]
plt.pie(size, labels=labels, colors=colors, explode=explode, autopct='%.2f%%', shadow=True, startangle=-30, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
plt.legend(labels, loc='upper right', title='Category')

plt.figure(figsize=(5, 5))
labels = ['Fin', 'RFn', 'Unf', 'NA']
size = data['GarageFinish'].value_counts()
colors = ['purple', 'lightblue', 'pink', 'yellow']
explode = [0, 0.1, 0, 0]
plt.pie(size, labels=labels, colors=colors, explode=explode, autopct='%.2f%%', shadow=True)
circle = plt.Circle((0, 0), 0.8, color='white')
p = plt.gcf()
p.gca().add_artist(circle)

plt.figure(figsize=(15, 9))
sns.countplot(x='CentralAir', hue='BedroomAbvGr', palette='terrain', data=data).set(title='BedroomAbvGr vs CentralAir')

sns.scatterplot(x='GarageArea', y='GarageYrBlt', data=data, color='lightcoral')

plt.figure(figsize=(20, 150), facecolor='white')
plotnumber = 1
for c in dtype_objects:
    ax = plt.subplot(20, 2, plotnumber)
    sns.barplot(x=data[c], y=data.SalePrice, palette='Set3')
    plotnumber += 1
    plt.xticks(rotation=70)

plt.figure(figsize=(15, 9))
splot = sns.barplot(x='Electrical', y='MSSubClass', hue='CentralAir', palette='nipy_spectral', data=data)

box = data[['LotArea', 'YearBuilt', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt']]
plt.figure(figsize=(15, 15), facecolor='white')
plotnum = 1
for c in box:
    if plotnum < 9:
        a = plt.subplot(4, 2, plotnum)
        sns.distplot(box[c])
    plotnum += 1
plt.tight_layout()
box = data[['LotArea', 'YearBuilt', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt']]
box
plt.figure(figsize=(10, 30), facecolor='white')
plotnumber = 1
for c in box:
    ax = plt.subplot(8, 1, plotnumber)
    sns.boxplot(data[c], color='green')
    plotnumber = plotnumber + 1

from scipy.stats import skew
numerical_features = data.dtypes[data.dtypes != 'object'].index
skewed_features = data[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'skew': skewed_features})
skewness
skewness = skewness[abs(skewness > 0.8)]
print('There are {} skewed numerical features to box cox transform'.format(skewness.shape[0]))
from scipy.special import boxcox1p
skewed_features = skewness.index
lamda = 0.15
for features in skewed_features:
    data[features] += 1
    data[features] = boxcox1p(data[features], lamda)
data[skewed_features] = np.log1p(data[skewed_features])
print('Skewness has been Handled using Box Cox Transformation')
data_object = data.select_dtypes(include='object').columns
print(data_object)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for features in data_object:
    data[features] = le.fit_transform(data[features].astype(str))
print(data.info())
x = data.drop(['Id', 'SalePrice'], axis=1)
y = data['SalePrice']
from sklearn.preprocessing import MinMaxScaler
mc = MinMaxScaler()
scaled_x = mc.fit_transform(x)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(scaled_x, y, test_size=0.2, random_state=0)
print('Traning Shape = ', x_train.shape)
print('Testing Shape = ', x_test.shape)
from sklearn.linear_model import LinearRegression
LR = LinearRegression()