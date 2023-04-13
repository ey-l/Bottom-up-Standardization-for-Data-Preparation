import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
cm = sns.light_palette('pink', as_cmap=True)
_input1.head(20).style.background_gradient(cmap=cm)
pd.DataFrame(_input1.dtypes, columns=['type'])
_input1.describe().style.background_gradient(cmap=cm)
isnull = _input1.isnull().sum().sort_values(ascending=False).to_frame()
isnull.columns = ['How_many']
isnull['precentage'] = np.around((isnull / len(_input1) * 100)[isnull / len(_input1) * 100 != 0], decimals=2)
isnull[isnull.How_many > 0].style.background_gradient(cmap=cm)
plt.figure(figsize=(35, 35))
sns.heatmap(_input1.corr(), annot=True, cmap='YlOrRd', linewidths=0.1, annot_kws={'fontsize': 10})
plt.title('Correlation house prices - return rate')
correlation = _input1.corr().unstack().sort_values(kind='quicksort', ascending=False)
correlation = correlation[correlation != 1]
print('Top 20 with highest positive correlation')
print(correlation[:20])
print('Top 20 with highest negative correlation')
print(correlation[-20:][::-1])
_input1 = _input1.dropna(axis=0, subset=['SalePrice'], inplace=False)
X_train_full = _input1.drop(['SalePrice'], axis=1)
y_train_full = _input1.SalePrice
(X_train, X_valid, y_train, y_valid) = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)
cat_columns = [column for column in X_train.columns if X_train[column].dtype == 'object']
cat_columns
num_columns = [column for column in X_train.columns if X_train[column].dtype != 'object']
num_columns
cat_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
num_trans = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())])
preprocessor = ColumnTransformer(transformers=[('num', num_trans, num_columns), ('cat', cat_trans, cat_columns)], remainder='drop')