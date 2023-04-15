import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.shape
train.shape
test.head()
train.columns
test.columns
train.SalePrice.describe()
plt.style.use('bmh')
sns.distplot(train['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4})
list(set(train.dtypes.tolist()))
train = train.drop(labels=['Id'], axis=1)
df_numerical = train.select_dtypes(include=['int64', 'float64'])
df_numerical.head()
corrmat = df_numerical.corr()
g = sns.heatmap(df_numerical.corr())
T_corr = corrmat.index[abs(corrmat['SalePrice']) > 0.55]
g = sns.heatmap(df_numerical[T_corr].corr(), annot=True, cmap='RdYlGn')
df_cat = train.select_dtypes(include=['O'])
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
scatter_plot(train.GrLivArea, train.SalePrice, 'GrLivArea vs SalePrice', 'GrLivArea', 'SalePrice', 10, 'Rainbow')
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
scatter_plot(train.GrLivArea, train.SalePrice, 'GrLivArea vs SalePrice', 'GrLivArea', 'SalePrice', 10, 'Rainbow')
scatter_plot(train.TotalBsmtSF, train.SalePrice, 'TotalBsmtSF Vs SalePrice', 'TotalBsmtSF', 'SalePrice', 10, 'Cividis')
train.drop(train[train.TotalBsmtSF > 3000].index, inplace=True)
train.reset_index(drop=True, inplace=True)
scatter_plot(train.TotalBsmtSF, train.SalePrice, 'TotalBsmtSF Vs SalePrice', 'TotalBsmtSF', 'SalePrice', 10, 'Cividis')
missing = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing, percent], axis=1, keys=['missing', 'percent'])
missing_data.head(20)
train.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'LotFrontage'], inplace=True)
train.shape
missing1 = test.isnull().sum().sort_values(ascending=False)
percent1 = (test.isnull().sum() / test.isnull().count()).sort_values(ascending=False)
missing_data1 = pd.concat([missing1, percent], axis=1, keys=['missing1', 'percent1'])
missing_data1.head(25)
test.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'LotFrontage'], inplace=True)
test.shape
train.fillna(method='ffill', inplace=True)
test.fillna(method='ffill', inplace=True)
df_cat = train.select_dtypes(include=['O'])
df_cat.head()
df_cat.columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1 = df_cat.apply(le.fit_transform)
df1.head(2)
df_numerical = train.select_dtypes(include=['int64', 'float64'])
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