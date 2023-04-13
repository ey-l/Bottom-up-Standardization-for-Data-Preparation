import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
print(_input1.shape)
_input1.head()
data_explore = _input1.copy()
data_explore = data_explore.drop(columns='Id', axis=1)
data_explore.info()
nulls = data_explore.isna().sum()
nulls[nulls > 0]
na_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageCond', 'GarageQual']
data_explore[na_cols] = data_explore[na_cols].fillna('NA')
data_explore['Alley'].value_counts()
nulls = data_explore.isna().sum()
nan_cols = nulls[nulls > 0].index
data_explore[nan_cols].info()
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')
num_nans = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
cat_nans = ['MasVnrType', 'Electrical', 'FireplaceQu']
data_explore[num_nans] = num_imputer.fit_transform(_input1[num_nans])
data_explore[cat_nans] = cat_imputer.fit_transform(_input1[cat_nans])
nulls = data_explore.isna().sum()
nan_cols = nulls[nulls > 0].index
nan_cols
data_explore.head()
data_explore['MSSubClass'] = data_explore['MSSubClass'].astype(str)
cat_attrs = []
num_attrs = []
columns = list(data_explore.columns)
for col in columns:
    if data_explore[col].dtype == 'O':
        cat_attrs.append(col)
    else:
        num_attrs.append(col)
data_explore.describe()
Q1 = data_explore.quantile(0.25)
Q3 = data_explore.quantile(0.75)
IQR = Q3 - Q1
outliers = ((data_explore < Q1 - 1.5 * IQR) | (data_explore > Q3 + 1.5 * IQR)).sum()
outliers[outliers > 0]
data_explore['SalePrice'].hist()
plt.hist(data_explore['SalePrice'].apply(np.log))
plt.figure(figsize=(85, 16))
corr_matrix = data_explore.corr()
sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), square=True, annot=True, cbar=False)
plt.tight_layout()
corr_matrix['SalePrice'].sort_values(ascending=False)
features_to_viz = ['GrLivArea', 'GarageArea', 'TotalBsmtSF']
i = 1
plt.style.use('seaborn')
plt.figure(figsize=(15, 6))
for feature in features_to_viz:
    plt.subplot(1, 3, i)
    i = i + 1
    plt.scatter(data_explore[feature], data_explore['SalePrice'])
    plt.title('Sale Price Vs ' + feature)
plt.figure(figsize=(10, 6))
sns.boxplot(x='OverallQual', y='SalePrice', data=data_explore)
plt.figure(figsize=(18, 8))
sns.boxplot(x='YearBuilt', y='SalePrice', data=data_explore)
plt.xticks(rotation=90)
plt.scatter(data_explore['GrLivArea'], data_explore['SalePrice'], c=data_explore['TotRmsAbvGrd'], cmap='Set2_r')
plt.title('SalePrice Vs. GrLivArea')
plt.colorbar().set_label('# of Total Rooms Above Ground', fontsize=14)
data_explore['GarageCars'].value_counts()
plt.scatter(data_explore['GrLivArea'], data_explore['SalePrice'], c=data_explore['GarageCars'], cmap='Set2_r')
plt.title('SalePrice Vs. GrLivArea')
plt.colorbar().set_label('Capacity of # Cars in Garage', fontsize=14)
plt.scatter(data_explore['GrLivArea'], data_explore['SalePrice'], c=data_explore['YearBuilt'].astype('int'), cmap='rainbow')
plt.title('SalePrice Vs. GrLivArea')
plt.colorbar().set_label('YearBuilt', fontsize=14)
features_to_viz = ['ExterQual', 'GarageQual', 'KitchenQual', 'FireplaceQu', 'BsmtQual', 'BsmtExposure']
i = 1
plt.figure(figsize=(15, 10))
for col in features_to_viz:
    plt.subplot(3, 2, i)
    sns.boxplot(y=col, x='SalePrice', data=data_explore, orient='h')
    i += 1
features_to_viz = ['BldgType', 'HouseStyle', 'Foundation', 'MSZoning']
i = 1
plt.figure(figsize=(15, 10))
for col in features_to_viz:
    plt.subplot(3, 2, i)
    sns.boxplot(y=col, x='SalePrice', data=data_explore, orient='h')
    i += 1
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y='SaleType', x='SalePrice', data=data_explore)
plt.subplot(1, 2, 2)
sns.boxplot(y='SaleCondition', x='SalePrice', data=data_explore)
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
print(cols)
sns.pairplot(data_explore[cols])
X = _input1.drop(columns=['SalePrice'], axis=1)
y = _input1['SalePrice'].copy()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
y_log_train = np.log(y_train)
y_log_test = np.log(y_test)
na_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageCond', 'GarageQual', 'PoolQC', 'Fence', 'MiscFeature']
cat_attrs = [cat for cat in cat_attrs if not cat in na_cols]
num_attrs.remove('SalePrice')
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('transformer', PowerTransformer(method='yeo-johnson', standardize=True))])
cat_pipeline_1 = Pipeline([('cat_na_fill', SimpleImputer(strategy='constant', fill_value='NA')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
cat_pipeline_2 = Pipeline([('cat_nan_fill', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
pre_process = ColumnTransformer([('drop_id', 'drop', ['Id']), ('cat_pipeline_1', cat_pipeline_1, na_cols), ('cat_pipeline_2', cat_pipeline_2, cat_attrs), ('num_pipeline', num_pipeline, num_attrs)], remainder='passthrough')
X_train_transformed = pre_process.fit_transform(X_train)
X_test_transformed = pre_process.transform(X_test)
(X_train_transformed.shape, X_test_transformed.shape)
oh_na_cols = list(pre_process.transformers_[1][1]['encoder'].get_feature_names(na_cols))
oh_nan_cols = list(pre_process.transformers_[2][1]['encoder'].get_feature_names(cat_attrs))
feature_columns = oh_na_cols + oh_nan_cols + num_attrs
from sklearn.model_selection import GridSearchCV, KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
from sklearn.linear_model import ElasticNet
elastic_net_grid_param = [{'l1_ratio': list(np.linspace(0, 1, 10)), 'alpha': [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1]}]
elastic_net_grid_search = GridSearchCV(ElasticNet(random_state=42), elastic_net_grid_param, cv=kf, scoring='neg_root_mean_squared_error', return_train_score=True, n_jobs=-1)