import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train.head()
train.shape
train.info()
plt.figure(figsize=(10, 10))
sns.heatmap(train.corr(), cmap='RdBu')
plt.title('Correlations Between Variables', size=15)

important_num_cols = list(train.corr()['SalePrice'][(train.corr()['SalePrice'] > 0.3) | (train.corr()['SalePrice'] < -0.3)].index)
cat_cols = ['MSZoning', 'Utilities', 'BldgType', 'Heating', 'KitchenQual', 'SaleCondition', 'LandSlope']
important_cols = important_num_cols + cat_cols
train_df = train[important_cols]
train_df.head()
print(train_df.isna().sum())
print('Total: ', train_df.isna().sum().sum())
sns.scatterplot(x='SalePrice', y='GarageYrBlt', data=train_df)
train['GarageYrBlt'].describe()
train_df['GarageYrBlt'] = pd.to_numeric(train_df['GarageYrBlt'])
train_df['GarageYrBlt'] = train_df['GarageYrBlt'].fillna(1)
bins = [0, 1900, 1950, 1970, 1990, 2010]
group_names = ['No_data', '1900-1949', '1950-1969', '1970-1989', '1990-2010']
train_df['GarageYrBlt'] = pd.cut(train_df['GarageYrBlt'], bins, labels=group_names, include_lowest=True)
train_df['GarageYrBlt'].value_counts()
train_df.isna().sum()
sns.barplot(x=train_df['GarageYrBlt'], y=train_df['SalePrice'], data=train_df)
train_df['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
train_df['LotFrontage'].isna().sum()
train_df.isna().sum()
train_df['MasVnrArea'].describe()
train_df.dtypes
sns.scatterplot(x=train_df['MasVnrArea'], y=train_df['SalePrice'], data=train_df)
bins = np.linspace(min(train_df['MasVnrArea']), max(train_df['MasVnrArea']), 9)
g_names = ['0-200', '200-400', '400-600', '600-800', '800-1000', '1000-1200', '1200-1400', '1400-1600']
train_df['MasVnrArea'] = pd.cut(train_df['MasVnrArea'], bins, labels=g_names, include_lowest=True)
train_df['MasVnrArea'].value_counts()
sns.barplot(x=train_df['MasVnrArea'], y=train_df['SalePrice'], data=train_df)

def get_vars_list_with_strong_correlation(dataframe):
    CorField = []
    CorrKoef = dataframe.corr()
    for column_index in CorrKoef:
        for var_index in CorrKoef.index[CorrKoef[column_index] > 0.9]:
            if column_index != var_index and var_index not in CorField and (column_index not in CorField):
                CorField.append(var_index)
                print('%s-->%s: r^2=%f' % (column_index, var_index, CorrKoef[column_index][CorrKoef.index == var_index].values[0]))
    return CorField
get_vars_list_with_strong_correlation(train_df)
sns.pairplot(train_df[important_cols])

def get_dataframe_with_dummy_vars(df, list_of_columns):
    for column in list_of_columns:
        dummy_var_df = pd.get_dummies(df[column])
        dummy_df = pd.DataFrame(df[column].value_counts())
        dummy_df.reset_index(inplace=True)
        for (index, value) in enumerate(dummy_df['index']):
            name = dummy_df.iloc[index, 0]
            new_column_name = column + '_' + name
            dummy_var_df.rename(columns={name: new_column_name}, inplace=True)
        df = pd.concat([df, dummy_var_df], axis=1)
        df.drop(column, axis=1, inplace=True)
    return df
list_of_cols = list(train_df.select_dtypes(['object', 'category']).columns)
print(list_of_cols)
train_df_d = get_dataframe_with_dummy_vars(train_df, list_of_cols)
train_df_d.head().T
X = train_df_d.drop('SalePrice', axis=1)
y = train_df_d['SalePrice']
important_num_cols.remove('SalePrice')
important_num_cols.remove('MasVnrArea')
important_num_cols.remove('GarageYrBlt')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[important_num_cols] = scaler.fit_transform(X[important_num_cols])
X.head()
from sklearn.model_selection import train_test_split, cross_val_score
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)).mean()
    return rmse

def evaluation(y, predictions):
    """Function Counts evaluation errors"""
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    return (mae, mse, rmse, r_squared)

def run_ml_and_get_performance_results(models_list):
    models = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score', 'RMSE (Cross-Validation)'])
    for model in models_list:
        print('-' * 30)
        str_model = str(model)
        print('Model ', str_model, ' Started')
        if 'Polynomial' in str_model:
            X_train_d = model.fit_transform(X_train)
            X_test_d = model.transform(X_test)
            lin_reg = LinearRegression()