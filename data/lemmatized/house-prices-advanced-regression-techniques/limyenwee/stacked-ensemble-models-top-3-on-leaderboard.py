import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from scipy.stats import skew, norm, probplot
import time
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge, HuberRegressor, LinearRegression
from sklearn.svm import SVR
from sklearn.cluster import KMeans
import catboost as cb
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
y = _input1['SalePrice']
_input1 = _input1.drop(['SalePrice'], axis=1)
_input1 = _input1.set_index('Id')
_input0 = _input0.set_index('Id')
null_list = []
for col in _input1.columns:
    null = _input1[col].isnull().sum()
    test_null = _input0[col].isnull().sum()
    if null != 0 or test_null != 0:
        null_list.append([col, null, test_null])
null_df = pd.DataFrame(null_list, columns=['Feature', 'Null', 'Test Null'])
null_df.set_index('Feature')
null_df['Total Null'] = null_df['Null'] + null_df['Test Null']
print('-------------------------')
print('Total columns with null:')
print(len(null_df))
print('-------------------------')
print('Total null values:')
print(null_df['Total Null'].sum(axis=0))
print('-------------------------')
sns.set_palette(sns.color_palette('pastel'))
sns.barplot(data=null_df.sort_values(by='Total Null', ascending=False).head(10), x='Feature', y='Total Null')
plt.xticks(rotation=70)
plt.title('Total Nulls in Feature')
full = pd.concat([_input1, _input0], axis=0).reset_index(drop=True)
null = _input0[_input0['MSZoning'].isnull()][['Neighborhood', 'MSZoning']]
plot_data = pd.concat([full[full['Neighborhood'] == 'IDOTRR'], full[full['Neighborhood'] == 'Mitchel']], axis=0)
sns.histplot(data=plot_data, x='MSZoning', hue='Neighborhood', multiple='dodge', shrink=0.9)
plt.title('Distribution of Zoning Classification')
_input0.loc[(_input0['Neighborhood'] == 'IDOTRR') & _input0['MSZoning'].isnull(), 'MSZoning'] = 'RM'
_input0.loc[(_input0['Neighborhood'] == 'Mitchel') & _input0['MSZoning'].isnull(), 'MSZoning'] = 'RL'
data = full[~full['LotFrontage'].isnull() & (full['LotFrontage'] <= 150) & (full['LotArea'] <= 20000)]
sns.lmplot(data=data, x='LotArea', y='LotFrontage', line_kws={'color': 'black'})
plt.ylabel('LotFrontage')
plt.xlabel('LotArea')
plt.title('LotArea vs LotFrontage')
area_vs_frontage = LinearRegression()
area_vs_frontage_X = data['LotArea'].values.reshape(-1, 1)
area_vs_frontage_y = data['LotFrontage'].values