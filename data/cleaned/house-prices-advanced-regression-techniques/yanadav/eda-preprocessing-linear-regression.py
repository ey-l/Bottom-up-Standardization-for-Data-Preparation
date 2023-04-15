import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
with open('_data/input/house-prices-advanced-regression-techniques/data_description.txt', encoding='utf8') as f:
    for line in f:
        print(line.strip())
train.head()
train.shape
train.info()
pd.options.display.max_rows = 100
train.nunique().sort_values(ascending=False)
plt.figure(figsize=(15, 8))
sns.heatmap(train.isnull(), cbar=False, cmap='gray')
missing_val = pd.DataFrame(train.isnull().sum()[train.isnull().sum() != 0].sort_values(ascending=False)).rename(columns={0: 'num_miss'})
missing_val['missing_perc'] = (missing_val / train.shape[0] * 100).round(1)
missing_val = missing_val.query('missing_perc > 40')
missing_val
drop_cols = missing_val.index.to_list()
drop_cols
train.drop(['Id'], axis=1, inplace=True)
train.drop(columns=drop_cols, axis=1, inplace=True)
train.shape
num_cols = train.select_dtypes(include=['number'])
cat_cols = train.select_dtypes(include=['object'])
print(f'The dataset contains {len(num_cols.columns.tolist())} numerical columns and {len(cat_cols.columns.tolist())} categorical columns')
num_cols.head()
num_cols.describe()
cat_cols.head()
cat_cols.describe()
num_corr_price = num_cols.corr()['SalePrice'][:-1]
num_corr_price
best_features = num_corr_price[abs(num_corr_price) > 0.35].sort_values(ascending=False)
print('There are {} strongly correlated numerical features with SalePrice:\n{}'.format(len(best_features), best_features))
for feature in best_features.index:
    num_corr_price.drop(feature, inplace=True)
for feature in num_corr_price.index:
    train.drop(feature, axis=1, inplace=True)
    num_cols.drop(feature, axis=1, inplace=True)
train.shape
num_corr = num_cols.corr()
corr_triu = num_corr.where(np.triu(np.ones(num_corr.shape), k=1).astype(np.bool))
plt.figure(figsize=(10, 10))
sns.heatmap(num_corr, annot=True, square=True, fmt='.2f', annot_kws={'size': 9}, mask=np.triu(corr_triu), cmap='coolwarm')
corr_triu_collinear = corr_triu.iloc[:-1, :-1]
collinear_features = [column for column in corr_triu_collinear.columns if any(corr_triu_collinear[column] > 0.6)]
train.drop(columns=collinear_features, inplace=True)
num_cols.drop(columns=collinear_features, inplace=True)
train.shape
plt.figure(figsize=(10, 10))
sns.heatmap(num_cols.corr(), annot=True, square=True, fmt='.2f', annot_kws={'size': 9}, mask=np.triu(num_cols.corr()), cmap='coolwarm')
num_cols.isna().sum()
num_cols['LotFrontage'].hist(bins=40)
num_cols['LotFrontage'].describe()
train['LotFrontage'].fillna(np.random.randint(59, 80), inplace=True)
train['LotFrontage'].isna().sum()
num_cols.MasVnrArea.hist(bins=50)
num_cols.MasVnrArea.fillna(0, inplace=True)
print('Number of features left in numerical features:', len(num_cols.columns))
print('Numerical Features left:')
print(num_cols.columns.values)
for i in range(0, len(num_cols.columns), 5):
    plt.figure(figsize=(15, 15))
    sns.pairplot(data=num_cols, x_vars=num_cols.columns[i:i + 5], y_vars=['SalePrice'])
train = train.drop(train.LotFrontage.sort_values(ascending=False)[:2].index)
train = train.drop(train.BsmtFinSF1.sort_values(ascending=False)[:1].index)
train = train.drop(train.MasVnrArea.sort_values(ascending=False)[:1].index)
train = train.drop(train.TotalBsmtSF.sort_values(ascending=False)[:1].index)
train = train.drop(train.GrLivArea.sort_values(ascending=False)[:2].index)
train.reset_index(drop=True, inplace=True)
train.shape
cat_cols_missing = cat_cols.columns[cat_cols.isnull().any()]
cat_cols_missing
imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
for feature in cat_cols_missing:
    cat_cols[feature] = imputer.fit_transform(cat_cols[feature].values.reshape(-1, 1))
    train[feature] = imputer.fit_transform(train[feature].values.reshape(-1, 1))
cat_cols.nunique().sort_values(ascending=False)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for feature in cat_cols.columns:
    cat_cols[feature] = le.fit_transform(cat_cols[feature])
    train[feature] = le.fit_transform(train[feature])
plt.figure(figsize=(15, 15))
sns.heatmap(cat_cols.corr(), square=True, mask=np.triu(cat_cols.corr()), cmap='coolwarm')
cat_corr = cat_cols.corr()
cat_corr_triu = cat_corr.where(np.triu(np.ones(cat_corr.shape), k=1).astype(np.bool))
cat_collinear_features = [column for column in cat_corr_triu.columns if any(cat_corr_triu[column] > 0.6)]
train.drop(columns=cat_collinear_features, inplace=True)
cat_cols.drop(columns=cat_collinear_features, inplace=True)
train.head()
train.replace([np.inf, -np.inf], np.nan)
train.isna().sum().sort_values(ascending=False)
train.MasVnrArea.describe()
train.MasVnrArea.fillna(0, inplace=True)
y = train['SalePrice']
X = train.iloc[:, :-1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
linreg = LinearRegression()