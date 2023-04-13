import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def exam_data_load(df, target, id_name='', null_name=''):
    if id_name == '':
        df = df.reset_index().rename(columns={'index': 'id'})
        id_name = 'id'
    else:
        id_name = id_name
    if null_name != '':
        df[df == null_name] = np.nan
    (X_train, X_test) = train_test_split(df, test_size=0.2, random_state=2021)
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[target])
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[target])
    return (X_train, X_test, y_train, y_test)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
(X_train, X_test, y_train, y_test) = exam_data_load(_input1, target='SalePrice', id_name='Id')
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
import pandas as pd
(X_train.shape, X_test.shape)
pd.set_option('display.max_columns', 100)
y_train['SalePrice'].hist()
y_test['SalePrice'].hist()
X_train.isnull().sum().sort_values(ascending=False)[:20]
X_test.isnull().sum().sort_values(ascending=False)[:20]
X_train.info()
X_train = X_train.select_dtypes(exclude=['object'])
X_test = X_test.select_dtypes(exclude=['object'])
target = y_train['SalePrice']
X_train_id = X_train.pop('Id')
X_test_id = X_test.pop('Id')
X_train.head(1)
from sklearn.impute import SimpleImputer
imp = SimpleImputer()
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
from sklearn.model_selection import train_test_split
(X_tr, X_val, y_tr, y_val) = train_test_split(X_train, target, test_size=0.15, random_state=2022)
(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)
from xgboost import XGBRegressor
model = XGBRegressor()