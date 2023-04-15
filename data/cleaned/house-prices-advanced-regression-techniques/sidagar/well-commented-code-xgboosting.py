import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import math
import seaborn as sns
import xgboost as xgb
import plotly.express as px
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_train.describe()
df_train.head()
for i in df_train.columns:
    print(i + ' \t: ' + str(df_train[i].isnull().sum()))
max_replacements = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'Electrical', 'SaleType']
zero_replacements = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea', 'MasVnrArea']
median_replacements = ['LotFrontage']
na_replacements = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType']
mean_replacements = ['GarageYrBlt']
for i in max_replacements:
    value = df_train[i].value_counts().idxmax()
    df_train[i] = df_train[i].fillna(value)
for i in median_replacements:
    value = df_train[i].median()
    df_train[i] = df_train[i].fillna(value)
for i in na_replacements:
    value = 'NA'
    df_train[i] = df_train[i].fillna(value)
for i in mean_replacements:
    value = df_train[i].mean()
    df_train[i] = df_train[i].fillna(value)
for i in zero_replacements:
    value = 0
    df_train[i] = df_train[i].fillna(value)
print('Total no. of null values now are : ' + str(df_train[i].isnull().sum().sum()))
list_cat = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
list_cont = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
print('No. of columns with categorical data values are : ' + str(len(list_cat)))
print('No. of columns with continuous data values are : ' + str(len(list_cont)))
df_train.head(10)
column = 'Id'
name = 'ID'
a = []
for i in df_train.index:
    a.append(name + ' : ' + str(df_train[column][i]))
df_train[column + '_visual'] = a
fig = px.box(data_frame=df_train.reset_index(), hover_name='Id_visual', y='SalePrice', hover_data=['MoSold', 'YrSold'], height=500, width=400, labels={'SalePrice': 'Sale Price in "$"'}, title='Box plot of the sale price(Hover for details)')
fig.show()
removed = 0
threshold = 450000
for i in df_train.index:
    if df_train['SalePrice'][i] > threshold:
        df_train = df_train.drop(i)
        removed += 1
print('Total data points removed till now are: ' + str(removed))
display_order = {}
for i in list_cat:
    a = []
    for j in df_train.groupby(i).mean().index:
        a.append(j)
    display_order[i] = a
display_order['Alley'] = ['Grvl', 'Pave', 'NA']
display_order['LandContour'] = ['Lvl', 'Bnk', 'HLS', 'Low']
display_order['LotConfig'] = ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3']
display_order['ExterQual'] = ['Ex', 'Gd', 'TA', 'Fa']
display_order['ExterCond'] = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
display_order['BsmtQual'] = ['Ex', 'Gd', 'TA', 'Fa', 'NA']
display_order['BsmtCond'] = ['Gd', 'TA', 'Fa', 'Po', 'NA']
display_order['BsmtExposure'] = ['Gd', 'Av', 'Mn', 'No', 'NA']
display_order['BsmtFinType1'] = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
display_order['BsmtFinType2'] = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
display_order['HeatingQC'] = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
display_order['Electrical'] = ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix']
display_order['KitchenQual'] = ['Ex', 'Gd', 'TA', 'Fa']
display_order['Functional'] = ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev']
display_order['FireplaceQu'] = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
display_order['GarageQual'] = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
display_order['GarageCond'] = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
display_order['GarageFinish'] = ['Fin', 'RFn', 'Unf', 'NA']
display_order['PoolQC'] = ['Ex', 'Gd', 'Fa', 'NA']
display_order['Fence'] = ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA']
display_order['SaleType'] = ['WD', 'CWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth']
display_order['SaleCondition'] = ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']
y = 'SalePrice'
n = 3
s = 20
(f, axes) = plt.subplots(19, n, figsize=(s, 6 * s))
counter = 0
for i in list_cat:
    sns.barplot(x=i, y=y, data=df_train, order=display_order[i], ax=axes[counter // n][counter % n], saturation=1)
    counter += 1
z = 1.96
x = 'Neighborhood'
df_temp = df_train.groupby(x).mean()
confidences = []
sale_visual = []
count = []
for i in df_temp.index:
    a = []
    counter = 0
    for j in df_train.index:
        if df_train[x][j] == i:
            a.append(df_train['SalePrice'][j] - df_temp['SalePrice'][i])
            counter += 1
    count.append(counter)
    std = np.std(a)
    confidence = std / math.sqrt(counter)
    confidences.append(z * confidence // 1)
    sale_visual.append('Sale Price : ' + str(df_temp['SalePrice'][i] // 1))
df_temp['Confidence'] = confidences
df_temp['sale_visual'] = sale_visual
df_temp['Total Count'] = count
count_per = []
for i in df_temp.index:
    per = df_temp['Total Count'][i] / np.sum(count)
    per = per * 10000 // 1
    per = per // 100
    count_per.append(str(per) + '%')
df_temp['Count Percentage'] = count_per
fig = px.bar(data_frame=df_temp.reset_index(), y='SalePrice', color=x, x=x, category_orders=display_order, error_y='Confidence', hover_name=sale_visual, opacity=1, hover_data=['Total Count', 'Count Percentage'], labels={y: 'Sale Price in "$"', 'Grvl': 'Gravel', 'Pave': 'Paved', 'NA': 'Not Paved'})
fig.show()
list_pure_categorical = ['MSSubClass', 'MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Foundation', 'Heating', 'GarageType', 'SaleType', 'SaleCondition', 'MiscFeature', 'MasVnrType']
categorical_ordered = ['Street', 'Alley', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']
list_continuous = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'LowQualFinSF', 'GrLivArea', '2ndFlrSF']
for i in categorical_ordered:
    a = []
    for j in df_train.index:
        for k in range(len(display_order[i])):
            if df_train[i][j] == display_order[i][k]:
                a.append(k + 1)
    df_train[i] = a
df_train
n = 3
s = 20
(f, axes) = plt.subplots(3 * n - 1, n, figsize=(s, 3 * s))
counter = 0
for i in list_cont:
    sns.boxplot(y=i, data=df_train, ax=axes[counter // n][counter % n])
    counter += 1
outlier = {'LotFrontage': 150, 'LotArea': 100000, 'MasVnrArea': 900, 'BsmtFinSF1': 2000, 'TotalBsmtSF': 2500, '1stFlrSF': 2500, 'GarageArea': 1130, 'WoodDeckSF': 600, 'OpenPorchSF': 310, 'EnclosedPorch': 310, '3SsnPorch': 350, 'MiscVal': 6000, 'GrLivArea': 3500, 'BsmtFullBath': 2.5, '2ndFlrSF': 1750}
for j in outlier:
    for i in df_train.index:
        if df_train[j][i] > outlier[j]:
            df_train = df_train.drop(i)
            removed += 1
for i in df_train.index:
    if df_train['YearBuilt'][i] < 1880:
        df_train = df_train.drop(i)
        removed += 1
print('Total data points removed till now are: ' + str(removed))
y = 'SalePrice'
n = 3
s = 20
(f, axes) = plt.subplots(8, n, figsize=(s, 3 * s), sharey=True)
counter = 0
for i in list_continuous:
    sns.lineplot(x=i, y=y, data=df_train, ax=axes[counter // n][counter % n])
    counter += 1
corr = df_train[categorical_ordered + list_continuous + ['SalePrice']].corr()
label = {'x': 'Column', 'y': 'Row', 'color': 'Correlation'}
columns = categorical_ordered + list_continuous + ['SalePrice']
"fig = px.imshow(img = corr, x = columns,y = columns,labels = label,\n                color_continuous_scale = [[0,'white'],[0.33,'yellow'],\n                                          [0.66,'red'],[1.0,'black']],\n                height = 1100,width = 1100,color_continuous_midpoint = 0,\n                title = 'Correlation matrix for continuous and ordered categorical data fields.')\nfig.show()"
columns = categorical_ordered + list_continuous + ['SalePrice']
useful = []
for i in columns:
    if corr[i]['SalePrice'] >= 0.15 or corr[i]['SalePrice'] <= -0.15:
        useful.append(i)
useful_category = []
for j in list_pure_categorical:
    for i in df_train.groupby(j).count().index:
        s = j + str(i)
        a = []
        for k in df_train.index:
            if df_train[j][k] == i:
                a.append(1)
            else:
                a.append(0)
        df_train[s] = a
        useful_category.append(s)
len(useful_category)
corr = df_train[useful_category + ['SalePrice']].corr()
label = {'x': 'Column', 'y': 'Row', 'color': 'Correlation'}
columns = useful_category + ['SalePrice']
"fig = px.imshow(img = corr, x = columns,y = columns,labels = label,\n                color_continuous_scale = [[0,'white'],[0.42,'yellow'],\n                                          [0.58,'red'],[1.0,'black']],\n                height = 1100,width = 1100,color_continuous_midpoint = 0,\n                title = 'Correlation matrix for one hot encoded categorical data fields.')\nfig.show()"
columns = useful_category + ['SalePrice']
final_useful = []
for i in columns:
    if corr[i]['SalePrice'] >= 0.15 or corr[i]['SalePrice'] <= -0.15:
        final_useful.append(i)
useful = useful + final_useful
useful
df_train_x = df_train[useful].drop(['SalePrice'], axis=1)
df_train_x.describe()
df_train_y = df_train[['SalePrice']]
df_train_y.describe()
(x_train, x_test, y_train, y_test) = train_test_split(df_train_x, df_train_y, test_size=0.1, random_state=42)
poly = PolynomialFeatures(degree=2)
poly_x_train = poly.fit_transform(x_train)
poly_x_test = poly.fit_transform(x_test)
xg = xgb.XGBRegressor(criterion='mse')
parameters = {'max_depth': [1, 2, 3, 4, 5, 6], 'eta': [0.01, 0.03, 0.05], 'alpha': [0], 'n_estimators': [100, 500, 800, 1000, 1200, 1400]}
models = ['Normal Linear Regression: ', 'Linear Regression over polynomial: ', 'Normal XGBoost: ', 'XGBoost over polynomial: ']
predict = []