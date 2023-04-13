import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import warnings
plt.style.use('seaborn-darkgrid')
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col=0)
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col=0)
_input1.dtypes.value_counts()
_input0.dtypes.value_counts()
null_value = pd.DataFrame()
null_value['variable'] = _input0.columns
null_value['Train'] = _input1.drop(labels='SalePrice', axis=1).isnull().sum().to_list()
null_value['Test'] = _input0.isnull().sum().to_list()
null_value
no_null_col = null_value[(null_value['Train'] == 0) & (null_value['Test'] == 0)].variable.to_list()
df_train = _input1[no_null_col]
df_train['SalePrice'] = _input1['SalePrice']
df_test = _input0[no_null_col]
df_train.describe().T

def plot_hist_box(data):
    (figure, axis) = plt.subplots(1, 2, figsize=(15, 5))
    plt.suptitle(data.name)
    axis[0].boxplot(data)
    axis[1] = sns.distplot(data)
    axis[1].lines[0].set_color('crimson')
    rect = plt.Rectangle((0, 0), 1, 1, fill=False, color='k', lw=3, transform=figure.transFigure, figure=figure)
    figure.patches.extend([rect])
plot_hist_box(df_train.SalePrice)
df_train['House_Age'] = df_train.YrSold - df_train.YearBuilt
df_train['RemodAge'] = df_train.YrSold - df_train.YearRemodAdd
df_train = df_train.drop(labels=['YrSold', 'YearBuilt', 'YearRemodAdd'], inplace=False, axis=1)
plot_hist_box(df_train.House_Age)
plot_hist_box(df_train.RemodAge)
df_test['House_Age'] = df_test.YrSold - df_test.YearBuilt
df_test['RemodAge'] = df_test.YrSold - df_test.YearRemodAdd
df_test = df_test.drop(labels=['YrSold', 'YearBuilt', 'YearRemodAdd'], inplace=False, axis=1)
df_train['Overall_Rating'] = (df_train.OverallCond + df_train.OverallQual) * 0.5
df_test['Overall_Rating'] = (df_test.OverallCond + df_test.OverallQual) * 0.5
df_train = df_train.drop(labels=['OverallQual', 'OverallCond'], inplace=False, axis=1)
df_test = df_test.drop(labels=['OverallQual', 'OverallCond'], inplace=False, axis=1)
plot_hist_box(df_train.Overall_Rating)
_input1.Street.isnull().sum()
df_train['LotFrontage'] = _input1.LotFrontage
df_test['LotFrontage'] = _input0.LotFrontage
imputer = KNNImputer(n_neighbors=2)
df_train.LotFrontage = imputer.fit_transform(df_train[['LotFrontage']])
df_test.LotFrontage = imputer.fit_transform(df_test[['LotFrontage']])
plot_hist_box(df_train.LotFrontage)
plot_hist_box(df_train.LotArea)
train_MS_label = set(df_train.MSSubClass.value_counts().index.tolist())
test_MS_label = set(df_test.MSSubClass.value_counts().index.tolist())
print(f'Additional data in test is {test_MS_label - train_MS_label}')
df_train = pd.get_dummies(df_train, columns=['MSSubClass'])
df_test = pd.get_dummies(df_test, columns=['MSSubClass'])
df_t_0_l = [0] * df_train.shape[0]
df_0 = pd.DataFrame(df_t_0_l, columns=['MSSubClass_150'])
df_train = df_train.join(df_0)
df_train.columns[df_train.dtypes != 'O']
from calendar import month_name
from collections import OrderedDict
x = lambda x: datetime.date(1900, x, 1).strftime('%B')
month = list(map(x, df_train.MoSold.value_counts().index.to_list()))
sale_count = df_train.MoSold.value_counts().to_list()
month_sale = dict(zip(month, sale_count))
monthj = list(month_name)
monthj.remove('')
month_sale = OrderedDict([(a, month_sale[a]) for a in monthj])
plt.figure(figsize=(15, 5))
colors = ['r', 'r', 'gray', 'gray', 'gray', 'g', 'g', 'gray', 'gray', 'gray', 'gray', 'gray']
plt.bar(month_sale.keys(), month_sale.values(), color=colors)
plt.title('Month-Wise House Sale')
plt.ylabel('Sale Count')
plt.xlabel('Month')
plt.xticks(rotation=45)
df_train = pd.get_dummies(df_train, columns=['MoSold'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['MoSold'], drop_first=True)
df_train = pd.get_dummies(df_train, columns=['SaleCondition'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['SaleCondition'], drop_first=True)
label = {'None': 0, 'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
df_train.ExterQual = df_train.ExterQual.fillna('None', inplace=False)
df_test.ExterQual = df_test.ExterQual.fillna('None', inplace=False)
df_train = df_train.replace({'ExterQual': label}, inplace=False)
df_test = df_test.replace({'ExterQual': label}, inplace=False)
df_train.ExterCond = df_train.ExterCond.fillna('None', inplace=False)
df_test.ExterCond = df_test.ExterCond.fillna('None', inplace=False)
df_train = df_train.replace({'ExterCond': label}, inplace=False)
df_test = df_test.replace({'ExterCond': label}, inplace=False)
df_train['External_QC'] = (df_train.ExterCond + df_train.ExterQual) * 0.5
df_test['External_QC'] = (df_test.ExterCond + df_test.ExterQual) * 0.5
df_train = df_train.drop(labels=['ExterQual', 'ExterCond'], inplace=False, axis=1)
df_test = df_test.drop(labels=['ExterQual', 'ExterCond'], inplace=False, axis=1)
plot_hist_box(df_train.External_QC)
df_train.Fireplaces = (df_train.Fireplaces > 0).astype(int)
plot_hist_box(df_train.Fireplaces)
df_test.Fireplaces = (df_test.Fireplaces > 0).astype(int)
plot_hist_box(df_test.Fireplaces)
df_train['Street'] = df_train['Street'].replace(['Pave', 'Grvl'], [0, 1], inplace=False)
df_test['Street'] = df_test['Street'].replace(['Pave', 'Grvl'], [0, 1], inplace=False)
plot_hist_box(df_train.Street)
dum_var = ['LotShape', 'LandContour', 'LandSlope', 'LotConfig', 'Neighborhood', 'RoofStyle', 'PavedDrive', 'Foundation']
df_train = pd.get_dummies(df_train, columns=dum_var, drop_first=True)
df_test = pd.get_dummies(df_test, columns=dum_var, drop_first=True)
label = {'None': 0, 'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
df_train.HeatingQC = df_train.HeatingQC.fillna('None', inplace=False)
df_test.HeatingQC = df_test.HeatingQC.fillna('None', inplace=False)
df_train = df_train.replace({'HeatingQC': label}, inplace=False)
df_test = df_test.replace({'HeatingQC': label}, inplace=False)
drop_obj = df_train.columns[df_train.dtypes == 'O'].to_list()
df_train = df_train.drop(labels=drop_obj, inplace=False, axis=1)
df_test = df_test.drop(labels=drop_obj, inplace=False, axis=1)
df_train = df_train.fillna(0, inplace=False)
df_train['Fence'] = _input1.Fence
df_test['Fence'] = _input0.Fence
df_train = df_train.fillna(0, inplace=False)
df_test = df_test.fillna(0, inplace=False)
df_train.Fence = (df_train.Fence != 0).astype(int)
df_test.Fence = (df_test.Fence != 0).astype(int)
X = df_train.drop('SalePrice', axis=1)
y = df_train[['SalePrice']]
print(X.shape)
print(y.shape)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
print('Training Data:{}'.format(X_train.shape[0]))
print('Testing Data:{}'.format(X_test.shape[0]))
Linear_Model = LinearRegression()