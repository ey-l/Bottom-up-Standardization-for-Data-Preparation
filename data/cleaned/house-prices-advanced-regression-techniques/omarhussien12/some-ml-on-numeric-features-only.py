import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
house_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
house_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test_id = house_test['Id']
house_train
Numlist1 = ['BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageQual', 'GarageCond']
Numlist2 = ['BsmtExposure']
Numlist3 = ['BsmtFinType1', 'BsmtFinType2']
Numlist4 = ['PoolQC']
Numlist5 = ['Fence']
Numlist6 = ['ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual']
Numlist7 = ['LotShape']
Numlist8 = ['LandSlope']
Numlist9 = ['Functional']
Numlist10 = ['GarageFinish']

def numeric_map1(x):
    return x.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, np.nan: 0})

def numeric_map2(y):
    return y.map({'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4, np.nan: 0})

def numeric_map3(z):
    return z.map({'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6, np.nan: 0})

def numeric_map4(a):
    return a.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4, np.nan: 0})

def numeric_map5(b):
    return b.map({'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4, np.nan: 0})

def numeric_map6(c):
    return c.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

def numeric_map7(d):
    return d.map({'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4})

def numeric_map8(e):
    return e.map({'Sev': 1, 'Mod': 2, 'Gtl': 3})

def numeric_map9(f):
    return f.map({'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8})

def numeric_map10(g):
    return g.map({'Unf': 1, 'RFn': 2, 'Fin': 3, np.nan: 0})
house_train[Numlist1] = house_train[Numlist1].apply(numeric_map1)
house_train[Numlist2] = house_train[Numlist2].apply(numeric_map2)
house_train[Numlist3] = house_train[Numlist3].apply(numeric_map3)
house_train[Numlist4] = house_train[Numlist4].apply(numeric_map4)
house_train[Numlist5] = house_train[Numlist5].apply(numeric_map5)
house_train[Numlist6] = house_train[Numlist6].apply(numeric_map6)
house_train[Numlist7] = house_train[Numlist7].apply(numeric_map7)
house_train[Numlist8] = house_train[Numlist8].apply(numeric_map8)
house_train[Numlist9] = house_train[Numlist9].apply(numeric_map9)
house_train[Numlist10] = house_train[Numlist10].apply(numeric_map10)
house_train
house_train = house_train.drop(['Id'], axis=1)
house_test = house_test.drop(['Id'], axis=1)
house_train.info()
columns = house_train.columns
type(list(columns))
house_train['Fence'].dtype != object
columns
numeric_columns = []
for column in columns:
    if house_train[column].dtype != object:
        numeric_columns.append(column)
numeric_columns
train_data = house_train[numeric_columns]
train_data.isnull().sum()
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].median())
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].median())
train_data['GarageYrBlt'] = train_data['GarageYrBlt'].fillna(train_data['GarageYrBlt'].median())
train_data.isnull().sum()
np.isfinite(train_data.any())
train_x = train_data.iloc[:, :-1]
train_y = train_data.iloc[:, -1]
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
(train_x, train_y) = shuffle(train_x, train_y, random_state=42)
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(train_x, train_y, test_size=0.25, random_state=42)
import seaborn as sns
sns.distplot(y_train)
y_train_l = np.log1p(y_train)
sns.distplot(y_train_l)
pd.DataFrame(y_train).head(3)
pd.DataFrame(y_train_l).head(3)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model1 = LinearRegression()