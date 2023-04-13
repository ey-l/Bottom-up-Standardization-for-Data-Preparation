import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.info()
_input1 = _input1.select_dtypes(include=np.number)
_input1.head()
_input1.info()
_input1[['GarageYrBlt', 'MasVnrArea', 'LotFrontage']].describe()
_input1[['GarageYrBlt', 'MasVnrArea', 'LotFrontage']].head()
_input1['MasVnrArea'].plot.kde()
_input1['MasVnrArea'].median()
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].median(), inplace=False)
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(_input1['GarageYrBlt'].median(), inplace=False)
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].median(), inplace=False)
_input1.info()
_input1[['GarageYrBlt', 'MasVnrArea', 'LotFrontage']].describe()
_input1.SalePrice.plot.kde()
np.log(_input1.SalePrice).plot.kde()
from sklearn.preprocessing import StandardScaler as SC
sc_x = SC()
x = np.array(_input1.drop(columns={'SalePrice', 'Id'}))
y = np.array(_input1['SalePrice']).reshape([1460, 1])
x = sc_x.fit_transform(x)
y = np.log1p(y)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.15, random_state=42)

def evaluate(model):
    score = r2_score(y_test, model.predict(x_test))
    rmse = mse(y_test, model.predict(x_test), squared=False)
    print('Test data results', rmse, score)
    score = r2_score(y_train, model.predict(x_train))
    rmse = mse(y_train, model.predict(x_train), squared=False)
    print('Train data results', rmse, score)
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
linreg = LR()