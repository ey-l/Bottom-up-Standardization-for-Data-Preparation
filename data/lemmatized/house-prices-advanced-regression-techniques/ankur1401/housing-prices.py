import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import classification_report, mean_squared_error, mean_squared_log_error
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1
_input0
Y = _input1['SalePrice']
data = pd.concat((_input1.drop('SalePrice', axis=1), _input0))
data = data.reset_index()
data = data.drop('index', axis=1)
data
data.info()
print(open('_data/input/house-prices-advanced-regression-techniques/data_description.txt').read())
corr_matrix = _input1.iloc[:, 1:].corr()
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(40, 40))
sns.heatmap(corr_matrix, annot=True, mask=mask)
plt.figure(figsize=(40, 20))
sns.heatmap(data.isna(), cmap='viridis', yticklabels=False)
for i in data[data['MSZoning'].isnull()]['MSSubClass'].values:
    mask = (data['MSSubClass'] == i) & (data['MSZoning'].isnull() == False)
    mask2 = (data['MSSubClass'] == i) & (data['MSZoning'].isnull() == True)
    idx = data[data[mask2].isnull()].index
    data.loc[idx, 'MSZoning'] = data[mask]['MSZoning'].mode().item()
plt.scatter(x=data['LotFrontage'], y=data['1stFlrSF'], s=2)
lin_reg = LinearRegression()