import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
plt.rcParams['figure.figsize'] = [16, 9]
_dir = '_data/input/house-prices-advanced-regression-techniques/'
train_df = pd.read_csv(_dir + 'train.csv', index_col=0)
train_df = train_df.assign(dataset='train')
test_df = pd.read_csv(_dir + 'test.csv', index_col=0)
test_df = test_df.assign(dataset='test')
df = pd.concat([train_df, test_df], axis=0)
dict_types = {c[0]: c[1] for c in zip(df.dtypes.index, df.dtypes)}
df.SalePrice = np.log(df.SalePrice)
remove_outliers_cols = ['LotArea', '1stFlrSF', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal']
z_score_threshold = 4.5
df = df.assign(outlier=False)
for c in remove_outliers_cols:
    df.loc[abs(scipy.stats.zscore(df[c])) > z_score_threshold, 'outlier'] = True
scale_cols = [c for c in df.dtypes[df.dtypes == float].keys() if c not in ['Id', 'SalePrice']]
scaler = StandardScaler()
for c in scale_cols:
    df[c] = scaler.fit_transform(df[[c]])
t_df = (df.isnull().sum() / df.shape[0]).to_frame()
t_df = t_df[t_df.iloc[:, 0] != 0]
t_df = t_df.assign(action='None')
t_df.loc[t_df.iloc[:, 0] >= 0.5, 'action'] = 'drop_col'
t_df.loc[t_df.iloc[:, 1] == 'None', 'action'] = 'med_mod'
cols_to_drop = [c for c in t_df[t_df.action == 'drop_col'].index if c not in ['SalePrice', 'dataset']]
cols_to_replace_with_med_or_mod = t_df[t_df.action == 'med_mod'].index
df = df.drop(cols_to_drop, axis=1)
for c in cols_to_replace_with_med_or_mod:
    if dict_types[c] in [int, float]:
        df.loc[df[c].isnull(), c] = df[c].median()
    elif dict_types[c] == object:
        df.loc[df[c].isnull(), c] = df[c].mode().values[0]
obj_cols = [c for c in df.dtypes[df.dtypes == object].keys() if c not in ['dataset']]
for c in obj_cols:
    means = df.iloc[:train_df.shape[0]].groupby(c).SalePrice.mean()
    df[c + '_mean_target'] = df[c].map(means)
df = df.drop(obj_cols, axis=1)
X = df[df.dataset == 'train'].drop(['SalePrice', 'dataset'], axis=1)
y = df[df.dataset == 'train']['SalePrice']
random_state = 40
test_size = 0.2
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=test_size, random_state=random_state)
model = linear_model.LinearRegression()