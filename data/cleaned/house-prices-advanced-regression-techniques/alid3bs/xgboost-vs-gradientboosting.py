import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
sns.set()
full_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
(full_data.shape, test_data.shape)
full_data.head()
full_data.info()
unusual_null_data = ['nan', 'NAN', 'NA', 'NULL', {}, [], '?', '.', '-', '_', '', ' ', '  ']
for column in full_data.columns:
    strange_null = np.array([x in unusual_null_data for x in full_data[column]])
    print(column, full_data[column].isna().sum(), strange_null.sum())
missing = full_data.isna().sum()
px.bar(missing[missing > 0].sort_values(), title='Null Values per feature')
px.box(full_data, y='SalePrice')
plt.rc('figure', figsize=(16, 8))
sns.histplot(full_data.SalePrice, kde=True)
full_data['SalePrice'].quantile([0.25, 0.75])
1.5 * (214000 - 129975) + 214000
full_data.drop(index=full_data[full_data['SalePrice'] >= 340000].index, inplace=True)
plt.rc('figure', figsize=(16, 8))
sns.histplot(full_data.SalePrice, kde=True)
X_full = full_data.copy()
y = full_data['SalePrice']
X_full.drop('SalePrice', axis=1, inplace=True)
numerical_cols = [cname for cname in X_full.columns if X_full[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X_full.columns if X_full[cname].nunique() < 10 and X_full[cname].dtype == 'object']
my_columns = numerical_cols + categorical_cols
X_full = X_full[my_columns].copy()
X_test = test_data[my_columns].copy()
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('std_scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='no_feature')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])
X_full = preprocessor.fit_transform(X_full)
X_test = preprocessor.transform(X_test)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(X_full, y, train_size=0.8, random_state=42)
learning_rate = np.arange(0.1, 0.5, 0.01)
RMSE_validation = []
RMSE_train = []
for i in learning_rate:
    model_GradBoos = GradientBoostingRegressor(n_estimators=60, random_state=32, max_depth=2, learning_rate=i)