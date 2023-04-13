import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input0.shape
_input1.shape
_input0.head()
_input1.columns
_input0.columns
_input1.SalePrice.describe()
plt.style.use('bmh')
sns.distplot(_input1['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4})
list(set(_input1.dtypes.tolist()))
_input1 = _input1.drop(labels=['Id'], axis=1)
df_numerical = _input1.select_dtypes(include=['int64', 'float64'])
df_numerical.head()
corrmat = df_numerical.corr()
g = sns.heatmap(df_numerical.corr())
T_corr = corrmat.index[abs(corrmat['SalePrice']) > 0.55]
g = sns.heatmap(df_numerical[T_corr].corr(), annot=True, cmap='RdYlGn')
df_cat = _input1.select_dtypes(include=['O'])
df_cat.head()
df_cat.columns
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)

def scatter_plot(x, y, title, xaxis, yaxis, size, c_scale):
    trace = go.Scatter(x=x, y=y, mode='markers', marker=dict(color=y, size=size, showscale=True, colorscale=c_scale))
    layout = go.Layout(hovermode='closest', title=title, xaxis=dict(title=xaxis), yaxis=dict(title=yaxis))
    fig = go.Figure(data=[trace], layout=layout)
    return iplot(fig)
scatter_plot(_input1.GrLivArea, _input1.SalePrice, 'GrLivArea vs SalePrice', 'GrLivArea', 'SalePrice', 10, 'Rainbow')
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 300000)].index)
scatter_plot(_input1.GrLivArea, _input1.SalePrice, 'GrLivArea vs SalePrice', 'GrLivArea', 'SalePrice', 10, 'Rainbow')
scatter_plot(_input1.TotalBsmtSF, _input1.SalePrice, 'TotalBsmtSF Vs SalePrice', 'TotalBsmtSF', 'SalePrice', 10, 'Cividis')
_input1 = _input1.drop(_input1[_input1.TotalBsmtSF > 3000].index, inplace=False)
_input1 = _input1.reset_index(drop=True, inplace=False)
scatter_plot(_input1.TotalBsmtSF, _input1.SalePrice, 'TotalBsmtSF Vs SalePrice', 'TotalBsmtSF', 'SalePrice', 10, 'Cividis')
missing = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing, percent], axis=1, keys=['missing', 'percent'])
missing_data.head(20)
_input1 = _input1.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'LotFrontage'], inplace=False)
_input1.shape
missing1 = _input0.isnull().sum().sort_values(ascending=False)
percent1 = (_input0.isnull().sum() / _input0.isnull().count()).sort_values(ascending=False)
missing_data1 = pd.concat([missing1, percent], axis=1, keys=['missing1', 'percent1'])
missing_data1.head(25)
_input0 = _input0.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'LotFrontage'], inplace=False)
_input0.shape
_input1 = _input1.fillna(method='ffill', inplace=False)
_input0 = _input0.fillna(method='ffill', inplace=False)
df_cat = _input1.select_dtypes(include=['O'])
df_cat.head()
df_cat.columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1 = df_cat.apply(le.fit_transform)
df1.head(2)
df_numerical = _input1.select_dtypes(include=['int64', 'float64'])
df_numerical.head()
data = pd.concat([df1, df_numerical], axis=1)
data.head()
corrmat1 = data.corr()
T1_corr = corrmat.index[abs(corrmat['SalePrice']) > 0.5]
g1 = sns.heatmap(data[T1_corr].corr(), annot=True, cmap='RdYlGn')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(x, y, train_size=0.8, random_state=44, shuffle=True)
LinearRegressionModel = LinearRegression()