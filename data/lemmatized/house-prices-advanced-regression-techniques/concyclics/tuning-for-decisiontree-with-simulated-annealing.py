import pandas as pd
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')

def checkNull_fillData(df):
    for col in df.columns:
        if len(df.loc[df[col].isnull() == True]) != 0:
            if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                df.loc[df[col].isnull() == True, col] = df[col].median()
            else:
                df.loc[df[col].isnull() == True, col] = 'Missing'
checkNull_fillData(_input1)
checkNull_fillData(_input0)
train_columns = []
for col in _input1.columns:
    if _input1[col].dtype == 'float64' or _input1[col].dtype == 'int64':
        train_columns.append(col)
train_columns
train_columns = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
X = _input1[train_columns]
y = _input1['SalePrice']
from sklearn.model_selection import train_test_split
(train_X, val_X, train_y, val_y) = train_test_split(X, y)
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, max_depth, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, max_depth=max_depth)