import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
house_df = _input1.copy()
house_df.head(3)
print('Train Data Shape:', house_df.shape)
print('\nTotal Feature type: \n', house_df.dtypes.value_counts())
corrmat = house_df.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(house_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
isnull_series = house_df.isnull().sum()
print('\nNull Column: \n', isnull_series[isnull_series > 0].sort_values(ascending=False))
house_df = house_df.fillna(house_df.mean(), inplace=False)
house_df = house_df.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=False)
house_df = house_df.dropna(axis=0, inplace=False)
plt.title('Original Price Histogram')
sns.distplot(house_df['SalePrice'])
plt.title('Log Transformed Sale Price Histogram')
log_SalePrice = np.log1p(house_df['SalePrice'])
sns.distplot(log_SalePrice)
original_SalePrice = house_df['SalePrice']
house_df['SalePrice'] = np.log1p(house_df['SalePrice'])
from scipy.stats import skew
features_index = house_df.dtypes[house_df.dtypes != 'object'].index
skew_features = house_df[features_index].apply(lambda x: skew(x))
skew_features_top = skew_features[skew_features > 1]
print(skew_features_top.sort_values(ascending=False))
house_df[skew_features_top.index] = np.log1p(house_df[skew_features_top.index])
house_df = house_df.drop(list(house_df.dtypes[house_df.dtypes == 'object'].index), axis=1, inplace=False)
plt.scatter(x=_input1['OverallQual'], y=_input1['SalePrice'])
plt.ylabel('SalePrice', fontsize=15)
plt.xlabel('OverallQual', fontsize=15)
plt.scatter(x=_input1['GrLivArea'], y=_input1['SalePrice'])
plt.ylabel('SalePrice', fontsize=15)
plt.xlabel('GrLivArea', fontsize=15)
cond1 = house_df['GrLivArea'] > np.log1p(4000)
cond2 = house_df['SalePrice'] < np.log1p(500000)
outlier_index = house_df[cond1 & cond2].index
house_df = house_df.drop(outlier_index, axis=0, inplace=False)
house_df.shape
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test_df = _input0.copy()
test_df.head(3)
test_df = test_df.set_index('Id', inplace=False)
test_df = test_df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=False)
test_df[skew_features_top.index] = np.log1p(test_df[skew_features_top.index])
test_df = test_df.drop(list(test_df.dtypes[test_df.dtypes == 'object'].index), axis=1, inplace=False)
test_df = test_df.fillna(0, inplace=False)
test_df.head()
test_df.shape
from sklearn.metrics import mean_squared_error

def get_rmse(model):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    print(model.__class__.__name__, np.round(rmse, 3))
    return rmse

def get_rmses(models):
    rmses = []
    for model in models:
        rmse = get_rmse(model)
        rmses.append(rmse)
    return rmses

def get_top_bottom_coef(model, n=10):
    coef = pd.Series(model.coef_, index=X_features.columns)
    coef_high = coef.sort_values(ascending=False).head(n)
    coef_low = coef.sort_values(ascending=False).tail(n)
    return (coef_high, coef_low)

def visualize_coefficient(models):
    (fig, axs) = plt.subplots(figsize=(24, 10), nrows=1, ncols=3)
    fig.tight_layout()
    for (i_num, model) in enumerate(models):
        (coef_high, coef_low) = get_top_bottom_coef(model)
        coef_concat = pd.concat([coef_high, coef_low])
        axs[i_num].set_title(model.__class__.__name__ + 'Coefficients', size=25)
        axs[i_num].tick_params(axis='y', direction='in', pad=-120)
        for label in axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels():
            label.set_fontsize(22)
        sns.barplot(x=coef_concat.values, y=coef_concat.index, ax=axs[i_num])

def get_top_features(model):
    ftr_importances_values = model.feature_importances_
    ftr_importances = pd.Series(ftr_importances_values, index=X_features.columns)
    ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
    return ftr_top20

def visualize_ftr_importances(models):
    (fig, axs) = plt.subplots(figsize=(24, 10), nrows=1, ncols=2)
    fig.tight_layout()
    for (i_num, model) in enumerate(models):
        ftr_top20 = get_top_features(model)
        axs[i_num].set_title(model.__class__.__name__ + ' Feature Importances', size=25)
        for label in axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels():
            label.set_fontsize(22)
        sns.barplot(x=ftr_top20.values, y=ftr_top20.index, ax=axs[i_num])
from sklearn.model_selection import cross_val_score

def get_avg_rmse_cv(models):
    for model in models:
        rmse_list = np.sqrt(-cross_val_score(model, X_features, y_target, scoring='neg_mean_squared_error', cv=5))
        rmse_avg = np.mean(rmse_list)
        print('\n{0} CV RMSE List: {1}'.format(model.__class__.__name__, np.round(rmse_list, 3)))
        print('{0} CV average RMSE: {1}'.format(model.__class__.__name__, np.round(rmse_avg, 3)))
from sklearn.model_selection import GridSearchCV

def get_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid=params, scoring='neg_mean_squared_error', cv=5)