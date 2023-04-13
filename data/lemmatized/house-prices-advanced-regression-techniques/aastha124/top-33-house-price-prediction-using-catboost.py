import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.describe()
_input1.skew()
_input1.skew().index[_input1.skew().values > 1]
outlier_cols = _input1.skew().index[_input1.skew().values > 1]
_input1.isnull().sum().index[_input1.isnull().sum().values > 0]
_input1.dtypes.unique()
df_copy = _input1.copy()
df_copy.shape
df_copy.select_dtypes(include=['O', 'object']).columns
cat_columns = df_copy.select_dtypes(include=['O', 'object']).columns
for cols in cat_columns:
    df_copy[cols] = df_copy[cols].fillna(df_copy[cols].mode()[0], inplace=False)
num_columns = df_copy.select_dtypes(exclude=['O', 'object']).columns
for cols in num_columns:
    if cols in outlier_cols:
        df_copy[cols] = df_copy[cols].fillna(df_copy[cols].median(), inplace=False)
    else:
        df_copy[cols] = df_copy[cols].fillna(df_copy[cols].mean(), inplace=False)
df_copy.isnull().sum().index[df_copy.isnull().sum().values > 0]
df_copy.skew()
import plotly.express as px
fig = px.box(df_copy['LotArea'])
fig.show('notebook')
fig = px.histogram(df_copy['LotArea'])
fig.show('notebook')
from scipy.stats.mstats import winsorize
px.histogram(winsorize(df_copy['LotArea'], limits=[0.05, 0.05]))
df_copy.corr()['SalePrice'].sort_values()
X = df_copy.copy()
X = X.drop(['SalePrice', 'Id'], axis=1, inplace=False)
y = df_copy['SalePrice']
px.histogram(y)
y = np.log1p(y)
px.histogram(y)
num_train_columns = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal', random_state=0)
X_num_transformed = qt.fit_transform(X[num_train_columns])
X[num_train_columns].shape
df_num_transformed = pd.DataFrame(X_num_transformed.reshape(-1, 36), columns=X[num_train_columns].columns)
df_num_transformed.head(1)
df_num_transformed.skew()
outlier_cols = df_num_transformed.skew().index[df_num_transformed.skew().values > 1]
for cols in outlier_cols:
    df_num_transformed[cols] = winsorize(df_num_transformed[cols], limits=[0.05, 0.05])
df_num_transformed.skew()
from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
X_num_transformed = rs.fit_transform(df_num_transformed)
df_num_transformed = pd.DataFrame(X_num_transformed.reshape(-1, 36), columns=X[num_train_columns].columns)
X_transformed = pd.concat([df_num_transformed, X[cat_columns]], axis=1)
X_transformed.head(1)
'\n### Label Encoding using JamesSteinEncoder\n\nThe correct way to implement encoders it to to do within cross-validation:\nhttp://kiwidamien.github.io/james-stein-encoder.html\n\nimport category_encoders as ce\n\n# Build the encoder\nencoder = ce.JamesSteinEncoder(cols=cat_columns)\n\n# Encode the frame and view it\nX_train_transformed_encoded = encoder.fit_transform(X_train_transformed, y_train)\n\n#transform the validation set\nX_val_transformed_encoded = encoder.transform(X_val_transformed, y_val)\n\n# Look at the first few rows\nX_train_transformed_encoded.head()\n'
import warnings
warnings.filterwarnings('once')
"\n!pip install xgboost\n\nfrom xgboost import XGBRegressor\nfrom sklearn.model_selection import cross_validate\nfrom sklearn.pipeline import Pipeline\nimport category_encoders as ce\nfrom sklearn.preprocessing import QuantileTransformer\n\n#we will use a pipeline to do the following \nqt=QuantileTransformer(output_distribution='normal', random_state=0)\nencoder = ce.JamesSteinEncoder(cols=cat_columns)\nxgb=XGBRegressor(random_state=123)\n\n# Build the model, including the encoder\nmodel = Pipeline([\n    ('james_stein', encoder), #categorical encoding\n    ('transformer', qt), #transformation to normal distribution\n    ('xgboost', xgb) #xgboost regressor\n])\n\nxgb_scores=cross_validate(model,X_train,y_train,cv=3,return_train_score=True,scoring=['neg_mean_squared_error','r2'])\n"
"\nfrom sklearn.model_selection import learning_curve\n\ntrain_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, X_train, y_train, cv=3,\n                                                                      return_times=True,random_state=123,\n                                                                     scoring='neg_mean_squared_error')\n    \ntrain_scores_mean = np.mean(train_scores, axis=1)\ntest_scores_mean = np.mean(test_scores, axis=1)\n\n\n# Plot learning curve\nimport plotly.graph_objects as go\n\n# Create traces\nfig = go.Figure()\nfig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean,\n                    mode='lines+markers',\n                    name='Training score'))\nfig.add_trace(go.Scatter(x=train_sizes, y=test_scores_mean,\n                    mode='lines+markers',\n                    name='Cross Validation score'))\n\nfig.show()\n"