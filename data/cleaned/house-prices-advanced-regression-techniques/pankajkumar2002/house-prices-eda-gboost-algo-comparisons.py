import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import plotly.figure_factory as ff
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import xgboost as xgb
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
raw_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
raw_data.head()
numerical_data = raw_data.select_dtypes(include=['int64', 'float64'])
categorical_data = raw_data.select_dtypes(include=['object'])
numerical_data.drop(['Id'], axis=1, inplace=True)
numerical_data.head()
numerical_data = numerical_data.astype('float64')
y_train = raw_data['SalePrice']
numerical_data.info()
numerical_data.describe().transpose()
numerical_data.isnull().sum()
corr = numerical_data.corr().round(2)
x = corr.index.to_list()
y = corr.columns.to_list()
c = []
for i in corr:
    c.append(list(corr[i].values))
hightlight_cols = list((corr['SalePrice'] > 0.2).index)[:-1]
numerical_data.drop(['SalePrice'], axis=1, inplace=True)
_template = dict(layout=go.Layout(font=dict(family='Franklin Gothic', size=14)))
fig = ff.create_annotated_heatmap(z=c, x=x, y=y, hovertemplate='Correlation between %{x} and %{y} = %{z} ', name='')
fig.update_layout(template='plotly_dark', paper_bgcolor='#2B2E4A', plot_bgcolor='#53354A', height=1000, margin=dict(l=120, t=150), yaxis=dict(showgrid=False))
fig.show()
numerical_data.tail(10)
fig = make_subplots(rows=len(hightlight_cols) // 3, cols=3)
for (i, col) in enumerate(hightlight_cols):
    _col = (i + 1) % 3
    _col = _col if _col != 0 else 3
    fig.add_trace(go.Box(y=numerical_data[col], name=col, boxpoints='all'), row=i // 3 + 1, col=_col)
fig.update_layout(template='plotly_dark', height=1800, showlegend=False, width=1500, title='Box Plot Distribution of correlated Features with Sale Price')
fig.show()
fig = make_subplots(rows=2, cols=3)
fig.add_trace(go.Histogram(x=numerical_data['OverallQual'], y=y_train, xbins=dict(start=-40, end=40, size=0.3), name='Overall'), row=1, col=1)
fig.add_trace(go.Scatter(x=numerical_data['GrLivArea'], y=y_train, name='GrLivArea', mode='markers', marker=dict(color='rgb(255, 25, 02)')), row=1, col=2)
fig.add_trace(go.Scatter(x=numerical_data['GarageArea'], y=y_train, name='GarageArea', mode='markers', marker=dict(color='rgb(86, 0, 222)')), row=1, col=3)
fig.add_trace(go.Scatter(x=numerical_data['GarageCars'], y=y_train, name='GarageArea', mode='markers'), row=2, col=1)
fig.add_trace(go.Scatter(x=numerical_data['FullBath'], y=y_train, name='GarageArea', mode='markers'), row=2, col=2)
fig.add_trace(go.Scatter(x=numerical_data['TotRmsAbvGrd'], y=y_train, name='GarageArea', mode='markers'), row=2, col=3)
fig.update_layout(template='plotly_dark', xaxis1_title='Overall', yaxis1_title='SalePrice', title='Max Correlated Features Distribution', xaxis2_title='GrLivArea', yaxis4_title='SalePrice', xaxis3_title='GarageArea', xaxis4_title='GarageCars', xaxis5_title='FullBath', xaxis6_title='TotRmsAbvGrd', showlegend=False, height=1000)
fig.show()
date_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
fig = go.Figure()
for col in date_cols:
    fig.add_trace(go.Scatter(x=raw_data[col], y=raw_data['SalePrice'], mode='markers', name=col))
fig.update_layout(template='plotly_dark', width=1300, title='SalePrice Distribution with Date columns', yaxis_title='SalePrice', xaxis_title='Year')
fig.show()
fig = make_subplots(rows=(categorical_data.shape[1] - 1) // 3, cols=3)
for (i, col) in enumerate(categorical_data.columns[:-1]):
    _col = (i + 1) % 3
    _col = _col if _col != 0 else 3
    series = raw_data.groupby(col)['SalePrice'].mean()
    fig.add_trace(go.Scatter(y=series, x=series.index, name=col), row=i // 3 + 1, col=_col)
    fig.update_yaxes(title='SalePrice')
fig.update_layout(template='plotly_dark', height=2000, width=1400)
fig.show()
categorical_data.describe()
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test_cat_col = list(test_data.select_dtypes(include=['object']))
test_data[test_cat_col].describe()

def ONEHOTCOLS(cat_df, test_df):
    one_hot_cols = []
    test_cat_col = list(test_data.select_dtypes(include=['object']))
    for idx in cat_df.describe():
        if cat_df.describe()[idx]['unique'] == test_df[test_cat_col].describe()[idx]['unique']:
            one_hot_cols.append(idx)
    return one_hot_cols

def OrdinalCols(cat_df, one_hot_cols):
    ordinal_cols = []
    for col in list(cat_df.columns):
        if col not in one_hot_cols:
            ordinal_cols.append(col)
    return ordinal_cols
one_hot_cols = ONEHOTCOLS(categorical_data, test_data)
ordinal_cols = OrdinalCols(categorical_data, one_hot_cols)
numeric_cols = list(numerical_data.columns)
numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaling', StandardScaler())])
one_hot_categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder())])
ordinal_categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder())])
preprocessing = ColumnTransformer(transformers=[('numeric', numerical_transformer, numeric_cols), ('cat', one_hot_categorical_transformer, one_hot_cols), ('cat2', ordinal_categorical_transformer, ordinal_cols)])
train_dataset = numerical_data.join(categorical_data)
x_train = preprocessing.fit_transform(train_dataset)
gb_model = GradientBoostingRegressor(random_state=42)
ada_model = AdaBoostRegressor(random_state=42)
rf_model = RandomForestRegressor(random_state=42)
xg_model = xgb.XGBRegressor()
models = [gb_model, ada_model, rf_model, xg_model]
model_name = ['GradientBoosting', 'AdaBoost', 'Random Forest', 'XGBoost']
for (i, model) in enumerate(models):
    score = cross_val_score(gb_model, x_train, y_train, cv=10)
    print('Accuracy of {} is {}'.format(model_name[i], score.mean()))