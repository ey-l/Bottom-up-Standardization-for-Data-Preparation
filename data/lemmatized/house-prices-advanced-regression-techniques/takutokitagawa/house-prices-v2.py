import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1.info()
_input1.head()
all_df = pd.concat([_input1.drop(columns='SalePrice'), _input0])
sns.histplot(_input1['SalePrice'])
num2str_list = ['MSSubClass', 'YrSold', 'MoSold']
for column in num2str_list:
    all_df[column] = all_df[column].astype(str)
all_df['YrSold'].head()
for column in all_df.columns:
    if all_df[column].dtype == 'O':
        all_df[column] = all_df[column].fillna('None')
    else:
        all_df[column] = all_df[column].fillna(0)
print(all_df.isnull().any())

def add_new_columns(df):
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
    df['AreaPerRoom'] = df['TotalSF'] / df['TotRmsAbvGrd']
    df['YearBuiltPlusRemod'] = df['YearBuilt'] + df['YearRemodAdd']
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df['Has2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
add_new_columns(all_df)
all_df = pd.get_dummies(all_df)
all_df.head()
all_df.info()
_input1.info()
_input1 = pd.merge(all_df.iloc[_input1.index[0]:_input1.index[-1]], _input1['SalePrice'], left_index=True, right_index=True)
_input0 = all_df.iloc[_input1.index[-1]:]
_input1.info()
_input1['SalePriceLog'] = np.log(_input1['SalePrice'])
sns.histplot(_input1['SalePriceLog'])
print(f"歪度: {round(_input1['SalePriceLog'].skew(), 4)}")
print(f"尖度: {round(_input1['SalePriceLog'].kurt(), 4)}")
train_X = _input1.drop(columns=['SalePrice', 'SalePriceLog'])
train_y = _input1['SalePriceLog']
test_X = _input1
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def lasso_tuning(train_x, train_y):
    param_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    for (cnt, alpha) in enumerate(param_list):
        lasso = Lasso(alpha=alpha)
        pipeline = make_pipeline(StandardScaler(), lasso)
        (X_train, X_test, y_train, y_test) = train_test_split(train_x, train_y, test_size=0.3, random_state=0)