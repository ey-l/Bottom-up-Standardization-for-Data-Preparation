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
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train_df.head()
train_df.info()
train_df = train_df.select_dtypes(include=np.number)
train_df.head()
train_df.info()
train_df[['GarageYrBlt', 'MasVnrArea', 'LotFrontage']].describe()
train_df[['GarageYrBlt', 'MasVnrArea', 'LotFrontage']].head()
train_df['MasVnrArea'].plot.kde()
train_df['MasVnrArea'].median()
train_df['MasVnrArea'].fillna(train_df['MasVnrArea'].median(), inplace=True)
train_df['GarageYrBlt'].fillna(train_df['GarageYrBlt'].median(), inplace=True)
train_df['LotFrontage'].fillna(train_df['LotFrontage'].median(), inplace=True)
train_df.info()
train_df[['GarageYrBlt', 'MasVnrArea', 'LotFrontage']].describe()
train_df.SalePrice.plot.kde()
np.log(train_df.SalePrice).plot.kde()
from sklearn.preprocessing import StandardScaler as SC
sc_x = SC()
x = np.array(train_df.drop(columns={'SalePrice', 'Id'}))
y = np.array(train_df['SalePrice']).reshape([1460, 1])
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